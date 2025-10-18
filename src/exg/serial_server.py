#!/usr/bin/env python3
import argparse
import logging
import time
from collections import deque
from collections.abc import Sequence

import numpy as np
import serial
from pylsl import local_clock

from .base_exg_server import BaseExgServer, resolve_serial_port

CHUNK_SAMPLES = 4        # push to LSL in small chunks
FALLBACK_FPS = 250.0       # used until we estimate from timestamps


class CircuitPySerialServer(BaseExgServer):
    def __init__(
        self,
        serial_port: str | None = None,
        baud: int = 115200,
        fps: float | None = None,
        lsl_raw: bool = True,
        lsl_filtered: bool = True,
        include_channels: list[int] | None = None,
        lowpass_fs: float | None = None,
        highpass_fs: float | None = None,
        notch_fs_list: list[float] | None = None,
    ) -> None:
        super().__init__(
            name="CircuitPySerialServer",
            fps=fps,
            include_channels=include_channels,
            lsl_raw=lsl_raw,
            lsl_filtered=lsl_filtered,
            lowpass_fs=lowpass_fs,
            highpass_fs=highpass_fs,
            notch_fs_list=notch_fs_list,
            source_id="CircuitPyLSL",
            daemon=True,
        )
        # Serial config
        self.serial_port = resolve_serial_port(
            serial_port,
            desc="CircuitPython",
            manu="Adafruit",
        )
        if not self.serial_port:
            raise RuntimeError("No serial port found. Pass --port explicitly.")
        self.baud = baud
        self.target_fps = fps  # if None, estimate from incoming data

        self.ser = None

        # Buffers
        self.raw_buf = deque(maxlen=4096)
        self.ts_buf = deque(maxlen=4096)

        # For dynamic FPS estimation
        self._last_arrival = None
        self._intervals = deque(maxlen=200)
        self._configured = False
        self._filter_fs = None

    # ---------- Setup ----------
    def _ensure_serial(self) -> None:
        if self.ser and self.ser.is_open:
            return
        self.ser = serial.Serial(self.serial_port, self.baud, timeout=1)
        self.log.info("Opened serial: %s @ %d", self.serial_port, self.baud)

    # ---------- Helpers ----------
    def _estimate_fps(self, now):
        if self._last_arrival is not None:
            self._intervals.append(now - self._last_arrival)
        self._last_arrival = now
        if self.target_fps is not None:
            return self.target_fps
        if len(self._intervals) >= 20:
            mean_dt = sum(self._intervals) / len(self._intervals)
            if mean_dt > 0:
                return 1.0 / mean_dt
        return FALLBACK_FPS

    def _parse_sample(self, line: bytes) -> list[float] | None:
        if not line:
            return None
        try:
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            return [float(part) for part in text.split(",")]
        except Exception:
            return None

    def _acquire_samples(self) -> tuple[np.ndarray, Sequence[float]] | None:
        self._ensure_serial()

        line = self.ser.readline()
        sample = self._parse_sample(line)
        if sample is None:
            return None

        latest_sample = sample
        latest_time = local_clock()

        # Drain any backlog so we always operate on the freshest data.
        while self.ser.in_waiting:
            next_line = self.ser.readline()
            parsed = self._parse_sample(next_line)
            if parsed is None:
                continue
            latest_sample = parsed
            latest_time = local_clock()

        fps_estimate = self._estimate_fps(latest_time)
        if not self._configured:
            fs = fps_estimate if fps_estimate is not None else FALLBACK_FPS
            self.configure_processing(sample_rate=fs, input_channel_count=len(latest_sample))
            self._configured = True
            self._filter_fs = fs
        elif (
            self._filter_fs is not None
            and abs(fps_estimate - self._filter_fs) / max(self._filter_fs, 1e-6) > 0.2
        ):
            self._filter_fs = fps_estimate
            self.update_filter(sample_rate=fps_estimate)

        if not self._configured:
            return None

        sample_np = np.asarray(latest_sample, dtype=np.float32)
        self.raw_buf.append(sample_np)
        self.ts_buf.append(latest_time)

        if len(self.raw_buf) >= CHUNK_SAMPLES:
            raw_chunk = np.stack(
                [self.raw_buf.popleft() for _ in range(CHUNK_SAMPLES)],
                axis=0,
            )
            ts_chunk = [self.ts_buf.popleft() for _ in range(CHUNK_SAMPLES)]
            return raw_chunk, ts_chunk

        return None

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.log.info("Serial closed. Exiting thread.")
        super().close()


def main():
    parser = argparse.ArgumentParser(description="CircuitPython Serial → LSL with IIR filtering")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--fps", type=float, default=None, help="Sampling rate hint (Hz); else estimate")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated channel indices to include (e.g. 0,2)")
    parser.add_argument("--no-raw", action="store_true", help="Disable raw LSL stream")
    parser.add_argument("--no-filtered", action="store_true", help="Disable filtered LSL stream")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    include_channels = None
    if args.include:
        include_channels = [int(x) for x in args.include.split(",")]

    server = CircuitPySerialServer(
        serial_port=args.port,
        baud=args.baud,
        fps=args.fps,
        lsl_raw=not args.no_raw,
        lsl_filtered=not args.no_filtered,
        include_channels=include_channels,
    )

    try:
        server.start()
        while server.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        server.stop()
        server.join()


if __name__ == "__main__":
    main()
