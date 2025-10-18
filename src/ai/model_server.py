from collections import deque
from typing import Optional

from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop, cf_float32
from PyQt5 import QtCore, QtCore as Qt
import numpy as np
import threading
import time
import os
import signal
import argparse

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, Input, BatchNormalization

from src.exg.filtering.ema import EMA


class ModelServer(QtCore.QThread):
    """LSL-based model trainer/inferencer running in a QThread.

    Modes:
      - "train": consumes EXG + prompt servers, updates a small Keras model.
      - "inference": consumes EXG and publishes a 4-dim prediction LSL stream.

    Signals:
      - mode_changed(str)
      - metrics_updated(float loss, float r2)
    """

    mode_changed = QtCore.pyqtSignal(str)
    metrics_updated = QtCore.pyqtSignal(float, float)

    def __init__(
        self,
        exg_stream_name: str = "filtered_exg",
        prompt_stream_name: str = "finger_prompt",
        model: Optional[tf.keras.Model] = None,
        batch_size: int = 1024,            # how many labeled samples per train burst
        ema_ms: int = 256,                 # EMA window if available; built
        buffer_length_s: float = 60.0,     # rolling buffer length in seconds
    ):
        super().__init__()

        # --- Config / state ---
        self.exg_stream_name = exg_stream_name
        self.prompt_stream_name = prompt_stream_name
        self.model = model
        self.running = True

        self._exg_inlet: Optional[StreamInlet] = None
        self._prompt_inlet: Optional[StreamInlet] = None
        self._output_outlet: Optional[StreamOutlet] = None
        self._output_nchannels: Optional[int] = None  # fixed to 4 once created

        self._mode = "train"
        self._mode_lock = threading.Lock()

        # Observed sampling rates & channel counts from stream infos
        self.exg_fs: Optional[float] = None
        self.prompt_fs: Optional[float] = None
        self.exg_ch: Optional[int] = None
        self.prompt_ch: Optional[int] = None

        # Rolling labeled sample buffers (allocated ONLY after exg_fs is known)
        self.buffer_length_s = float(buffer_length_s)
        self._train_batch_target = int(max(1, batch_size))
        self._last_train_time = 0.0
        self._min_train_interval_s = 0.25  # avoid training too often

        self._X: Optional[deque[np.ndarray]] = None
        self._y: Optional[deque[np.ndarray]] = None
        self._buf_max_samples: Optional[int] = None

        # Filters: build AFTER we know true channel count and exg sampling rate
        self.ema_ms = int(max(0, ema_ms))
        self.ema = None
        self._ema_channels: Optional[int] = None

        # Last computed metrics (for UI display)
        self._last_loss: Optional[float] = None
        self._last_r2: Optional[float] = None

        # Save request handled on the server thread
        self._save_request: Optional[str] = None

        # lightweight backoff to re-resolve missing streams at runtime
        self._last_resolve_attempt = 0.0
        self._resolve_backoff_s = 0.5

    # ----------------------
    # Public controls
    # ----------------------
    def set_mode(self, mode: str) -> None:
        if mode not in ("train", "inference"):
            raise ValueError("mode must be 'train' or 'inference'")
        with self._mode_lock:
            self._mode = mode
        self.mode_changed.emit(mode)

    def toggle_mode(self) -> None:
        with self._mode_lock:
            new_mode = "inference" if self._mode == "train" else "train"
        self.set_mode(new_mode)

    def stop(self) -> None:
        self.running = False

    def request_save(self, path: Optional[str] = None) -> None:
        """Request the server thread to save the current model to `path`.
        The actual write happens in the server thread to avoid blocking the UI.
        """
        self._save_request = path

    # ----------------------
    # Internals
    # ----------------------
    def _current_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _resolve_inlet(self, name: str, timeout: float = 0.2) -> Optional[StreamInlet]:
        try:
            results = resolve_byprop("name", name, timeout=timeout)
            return StreamInlet(results[0]) if results else None
        except Exception:
            return None

    def _ensure_outlet(self, n_channels: int) -> None:
        """Create/recreate the prediction outlet if needed."""
        try:
            if self._output_outlet is not None and self._output_nchannels == n_channels:
                return
            srate = float(self.exg_fs) if self.exg_fs else 0.0
            info = StreamInfo("model_output", "Pred", n_channels, srate, cf_float32, "model_outlet")
            self._output_outlet = StreamOutlet(info)
            self._output_nchannels = n_channels
        except Exception:
            self._output_outlet = None
            self._output_nchannels = None

    def _ensure_model(self, in_feats: int, out_feats: int = 4) -> None:
        if self.model is not None:
            return
        model = Sequential(
            [
                Input(shape=(in_feats,)),
                GaussianNoise(0.05),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.25),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.25),
                Dense(out_feats, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        self.model = model

    def _ensure_buffers(self) -> None:
        """Allocate/resize rolling buffers once exg_fs is known. No preallocation."""
        if not self.exg_fs or self.exg_fs <= 0:
            return
        target_len = max(1, int(self.buffer_length_s * float(self.exg_fs)))

        if self._buf_max_samples == target_len and self._X is not None and self._y is not None:
            return

        if self._X is None or self._y is None:
            # First-time allocation
            self._X = deque(maxlen=target_len)
            self._y = deque(maxlen=target_len)
        else:
            # Resize while preserving tail
            X_tail = list(self._X)[-target_len:]
            Y_tail = list(self._y)[-target_len:]
            self._X = deque(X_tail, maxlen=target_len)
            self._y = deque(Y_tail, maxlen=target_len)

        self._buf_max_samples = target_len
        print(f"[ModelServer] Buffer initialized/resized to {target_len} samples (fs={self.exg_fs} Hz)")

    def _ensure_filter(self, exg_channels: int) -> None:
        """Instantiate (or re-instantiate) the EMA after we know true channel count and exg fs."""
        if EMA is None or self.ema_ms <= 0 or not self.exg_fs or self.exg_fs <= 0:
            self.ema = None
            self._ema_channels = None
            return
        if self.ema is not None and self._ema_channels == exg_channels:
            return
        try:
            self.ema = EMA(window_intervals_ms=[self.ema_ms], num_channels=exg_channels, fs=int(self.exg_fs), methods=['mean'])
            self._ema_channels = exg_channels
            print(f"[ModelServer] EMA initialized: ch={exg_channels}, window_ms={self.ema_ms}, fs={self.exg_fs}")
        except Exception as exc:
            print(f"[ModelServer] EMA init failed, continuing without EMA: {exc}")
            self.ema = None
            self._ema_channels = None

    @staticmethod
    def _map_prompt_to_label(raw: np.ndarray) -> np.ndarray:
        """Map incoming prompt vector (4 or 5) to 4-output label (thumb,index,middle,ring_or_pinky)."""
        try:
            if raw.size >= 5:
                mapped = np.asarray([raw[0], raw[1], raw[2], max(raw[3], raw[4])], dtype=np.float32)
            elif raw.size == 4:
                mapped = np.asarray([raw[0], raw[1], raw[2], raw[3]], dtype=np.float32)
            else:
                tmp = list(raw) + [0.0] * (4 - raw.size)
                mapped = np.asarray(tmp[:4], dtype=np.float32)
        except Exception:
            mapped = np.zeros(4, dtype=np.float32)
        return mapped

    def _filter(self, vec: np.ndarray) -> np.ndarray:
        if self.ema is None:
            return vec
        try:
            return self.ema.process(vec.reshape(1, -1)).reshape(-1)
        except Exception:
            return vec

    def _train_incremental(self) -> None:
        """Train briefly (or evaluate) on a mini-batch without blocking the stream loop."""
        if self.model is None or self._X is None or len(self._X) == 0:
            return
        now = time.time()
        if (now - self._last_train_time) < self._min_train_interval_s:
            return

        n = min(len(self._X), self._train_batch_target)
        X_np = np.asarray(list(self._X)[-n:], dtype=np.float32)
        y_np = np.asarray(list(self._y)[-n:], dtype=np.float32)
        if X_np.size == 0 or y_np.size == 0:
            return

        try:
            loss_val: Optional[float] = None
            preds = None
            if X_np.shape[0] >= self._train_batch_target:
                res = self.model.train_on_batch(X_np, y_np)
                self._last_train_time = now
                if isinstance(res, dict):
                    loss_val = float(res.get("loss", next(iter(res.values()))))
                else:
                    try:
                        loss_val = float(res)
                    except Exception:
                        loss_val = None
            else:
                preds = self.model.predict(X_np, verbose=0)
                bcel = tf.keras.losses.binary_crossentropy(y_np, preds).numpy()
                loss_val = float(np.mean(bcel))

            # R2 on per-sample means (simple, stable)
            if preds is None:
                preds = self.model.predict(X_np, verbose=0)
            y_mean = y_np.mean(axis=1)
            p_mean = np.asarray(preds, dtype=np.float32).mean(axis=1)
            ss_res = float(np.sum((y_mean - p_mean) ** 2))
            ss_tot = float(np.sum((y_mean - float(np.mean(y_mean))) ** 2))
            r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            if loss_val is not None:
                self._last_loss = loss_val
            self._last_r2 = r2_val

            loss_emit = float(self._last_loss) if self._last_loss is not None else float("nan")
            r2_emit = float(self._last_r2) if self._last_r2 is not None else float("nan")
            print(f"[ModelServer] metrics -> loss={loss_emit:.6f}, r2={r2_emit:.6f}")
            self.metrics_updated.emit(loss_emit, r2_emit)
        except Exception:
            pass

    def _prepare_inlets(self) -> None:
        """Resolve LSL inlets and record nominal sampling rates and channel counts."""
        self._exg_inlet = self._resolve_inlet(self.exg_stream_name)
        self._prompt_inlet = self._resolve_inlet(self.prompt_stream_name)

        if self._exg_inlet is None:
            print(f"[ModelServer] EXG stream '{self.exg_stream_name}' not found (yet).")
        if self._prompt_inlet is None:
            print(f"[ModelServer] Prompt server '{self.prompt_stream_name}' not found; training will be skipped.")

        try:
            if self._exg_inlet is not None:
                inf = self._exg_inlet.info()
                fs = float(inf.nominal_srate())
                self.exg_fs = fs if fs > 0 else None
                ch = int(inf.channel_count())
                self.exg_ch = ch if ch > 0 else None
        except Exception:
            self.exg_fs = None
            self.exg_ch = None

        try:
            if self._prompt_inlet is not None:
                pinf = self._prompt_inlet.info()
                pfs = float(pinf.nominal_srate())
                self.prompt_fs = pfs if pfs > 0 else None
                pch = int(pinf.channel_count())
                self.prompt_ch = pch if pch > 0 else None
        except Exception:
            self.prompt_fs = None
            self.prompt_ch = None

        # If fs is known already, allocate buffers now; otherwise defer until first sample.
        self._ensure_buffers()

        # If we already know channels & fs, we can prebuild the filter.
        if self.exg_ch and self.exg_fs:
            self._ensure_filter(self.exg_ch)

    # ----------------------
    # Thread body
    # ----------------------
    def run(self) -> None:
        OUT_CH = 4

        self._prepare_inlets()

        # Defer model/outlet creation until EXG inlet is available & we know channel count from its StreamInfo
        while self.running:
            mode = self._current_mode()

            # Pull EXG sample
            exg = None
            if self._exg_inlet is not None:
                try:
                    sample, _ = self._exg_inlet.pull_sample(timeout=0.02)
                    if sample is not None:

                        # Ensure buffers and filter if we now have fs/ch from the inlet info
                        self._ensure_buffers()
                        if self.exg_ch and self.exg_fs:
                            if self.ema is None and self.ema_ms > 0:
                                self._ensure_filter(self.exg_ch)
                            if self.model is None:
                                self._ensure_model(in_feats=self.exg_ch, out_feats=OUT_CH)
                            if self._output_outlet is None:
                                self._ensure_outlet(n_channels=OUT_CH)

                        exg = np.asarray(sample, dtype=np.float32)
                        exg = self._filter(exg)
                except Exception:
                    exg = None

            if mode == "train":
                if exg is not None and self._prompt_inlet is not None and self._X is not None:
                    try:
                        p_sample, _ = self._prompt_inlet.pull_sample(timeout=0.0)
                    except Exception:
                        p_sample = None

                    if p_sample is not None:
                        mapped = self._map_prompt_to_label(np.asarray(p_sample, dtype=np.float32))
                        self._X.append(exg)
                        self._y.append(mapped)
                        self._train_incremental()

            else:  # inference
                if exg is not None and self.model is not None:
                    try:
                        pred = self.model.predict(exg.reshape(1, -1), verbose=0)
                        pred_flat = pred.reshape(-1).astype(np.float32)
                    except Exception:
                        pred_flat = np.zeros(self._output_nchannels or OUT_CH, dtype=np.float32)

                    try:
                        if self._output_outlet is not None:
                            out_n = self._output_nchannels or OUT_CH
                            vec = list(pred_flat[:out_n]) + [0.0] * max(0, out_n - len(pred_flat))
                            self._output_outlet.push_sample([float(v) for v in vec])
                    except Exception:
                        pass

            # Give up timeslice if no data
            time.sleep(0.002)

            # Handle save requests (non-blocking)
            if self._save_request is not None:
                try:
                    path = self._save_request or f"model_{int(time.time())}.keras"
                    d = os.path.dirname(path)
                    if d and not os.path.exists(d):
                        os.makedirs(d, exist_ok=True)
                    if self.model is not None:
                        self.model.save(path)
                        print(f"[ModelServer] Model saved to {path}")
                except Exception as exc:
                    print(f"[ModelServer] Failed to save model: {exc}")
                finally:
                    self._save_request = None

        # Cleanup at thread exit
        try:
            if self._exg_inlet is not None:
                self._exg_inlet.close_stream()
        except Exception:
            pass
        try:
            if self._prompt_inlet is not None:
                self._prompt_inlet.close_stream()
        except Exception:
            pass

# ----------------------
# Minimal runnable example
# ----------------------

def _print_metrics(loss: float, r2: float):
    print(f"[Metrics] loss={loss:.6f}, r2={r2:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run the ModelServer LSL trainer/inferencer")
    parser.add_argument("--exg", default="filtered_exg", help="LSL stream name for EXG samples")
    parser.add_argument("--prompt", default="finger_prompt", help="LSL stream name for prompt labels")
    parser.add_argument("--mode", choices=["train", "inference"], default="train", help="Initial mode")
    parser.add_argument("--ema_ms", type=int, default=256, help="EMA window (ms); 0 disables EMA")
    parser.add_argument("--buffer_s", type=float, default=30.0, help="Rolling buffer length (seconds)")
    parser.add_argument("--batch", type=int, default=1024, help="Train batch size in samples")

    args = parser.parse_args()

    # Create a Qt Core app (no GUI needed) so QThread has an event loop
    app = Qt.QCoreApplication([])

    server = ModelServer(
        exg_stream_name=args.exg,
        prompt_stream_name=args.prompt,
        batch_size=args.batch,
        ema_ms=max(0, args.ema_ms),
        buffer_length_s=args.buffer_s,
    )

    server.mode_changed.connect(lambda m: print(f"[ModelServer] mode -> {m}"))
    server.metrics_updated.connect(_print_metrics)
    server.set_mode(args.mode)
    server.start()

    def _graceful_exit(*_):
        print("[ModelServer] Stopping...")
        server.stop()
        server.wait(2000)
        app.quit()

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    app.exec_()


if __name__ == "__main__":
    main()