#!/usr/bin/env python3
import sys
import time
import logging
import threading
import argparse
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
from pylsl import StreamInfo, StreamOutlet, local_clock

from iir import IIR  # your existing filter

DEFAULT_BAUD = 115200
CHUNK_SAMPLES = 4        # push to LSL in small chunks
FALLBACK_FPS = 250.0       # used until we estimate from timestamps

def find_circuitpy_port() -> str | None:
    """
    Try to locate the CircuitPython USB-CDC serial port.
    Preference order:
      - desc/manufacturer mentions 'CircuitPython' or 'Adafruit'
      - macOS usbmodem
      - generic USB serial candidates
    """
    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = f"{p.description}".lower()
        manu = f"{p.manufacturer}".lower() if p.manufacturer else ""
        if "circuitpython" in desc or "circuitpython" in manu or "adafruit" in manu:
            return p.device
        if "usbmodem" in p.device.lower():
            candidates.append(p.device)
        elif "usbserial" in p.device.lower():
            candidates.append(p.device)
    return candidates[0] if candidates else None


class CircuitPySerialServer(threading.Thread):
    def __init__(self, port:str|None=None, baud:int=DEFAULT_BAUD, fps:float|None=None,
                 lsl_raw:bool=True, lsl_filtered:bool=True, include_channels:list[int]|None=None):
        super().__init__(daemon=True)
        self.log = logging.getLogger("CircuitPySerialServer")
        self.port = port or find_circuitpy_port()
        if not self.port:
            raise RuntimeError("No CircuitPython serial port found. Pass --port explicitly.")
        self.baud = baud
        self.target_fps = fps  # if None, estimate from incoming data
        self.include_channels = include_channels
        self.lsl_raw = lsl_raw
        self.lsl_filtered = lsl_filtered

        self.ser = None
        self.running = False

        # will be set after first line is parsed
        self.num_channels = None
        self.fs_for_filter = None
        self.iir = None

        self.raw_outlet = None
        self.filt_outlet = None

        # Buffers
        self.raw_buf = deque(maxlen=4096)
        self.ts_buf = deque(maxlen=4096)

        # For dynamic FPS estimation
        self._last_arrival = None
        self._intervals = deque(maxlen=200)

    # ---------- Setup ----------
    def open_serial(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        self.log.info(f"Opened serial: {self.port} @ {self.baud}")

    def init_lsl_outlets(self):
        ch_count = self.num_channels
        # Names match your BrainFlow version for drop-in use in LSL viewers
        if self.lsl_raw:
            info = StreamInfo(
                name="raw_exg",
                type="EXG",
                channel_count=ch_count,
                nominal_srate=self.fs_for_filter,
                channel_format="float32",
                source_id="CircuitPyLSL"
            )
            self.raw_outlet = StreamOutlet(info)
            self.log.info("LSL raw_exg outlet created.")
        if self.lsl_filtered:
            infof = StreamInfo(
                name="filtered_exg",
                type="EXG",
                channel_count=ch_count,
                nominal_srate=self.fs_for_filter,
                channel_format="float32",
                source_id="CircuitPyLSL"
            )
            self.filt_outlet = StreamOutlet(infof)
            self.log.info("LSL filtered_exg outlet created.")

    def init_filter(self):
        ch = self.num_channels
        fs = float(self.fs_for_filter)
        # Keep your original choices: HP=10 Hz, LP=Nyquist, Notch 50/60
        self.iir = IIR(
            num_channels=ch,
            fs=fs,
            lowpass_fs=fs/2.0,
            highpass_fs=15.0,
            notch_fs_list=[60],
            filter_order=4
        )
        self.log.info(f"IIR initialized with fs={fs:.3f} Hz, channels={ch}")

    # ---------- Helpers ----------
    def _estimate_fps(self, now):
        if self._last_arrival is not None:
            self._intervals.append(now - self._last_arrival)
        self._last_arrival = now
        if self.target_fps is not None:
            return self.target_fps
        if len(self._intervals) >= 20:
            mean_dt = sum(self._intervals)/len(self._intervals)
            if mean_dt > 0:
                return 1.0/mean_dt
        return FALLBACK_FPS

    def _select_channels(self, arr:np.ndarray) -> np.ndarray:
        if self.include_channels:
            return arr[:, self.include_channels]
        return arr

    def _maybe_init_after_first_line(self, sample:list[float], fps_hint:float):
        if self.num_channels is None:
            self.num_channels = len(sample) if not self.include_channels else len(self.include_channels)
            self.fs_for_filter = fps_hint if fps_hint is not None else FALLBACK_FPS
            self.init_lsl_outlets()
            self.init_filter()

    # ---------- Main loop ----------
    def run(self):
        self.running = True
        self.open_serial()
        self.log.info("Start reading serial and streaming to LSL.")
        try:
            while self.running:
                line = self.ser.readline()
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    # Accept CSV or single integer
                    parts = text.split(",")
                    sample = [float(p) for p in parts]
                except Exception:
                    # skip malformed line
                    continue

                now = local_clock()
                fps = self._estimate_fps(now)
                self._maybe_init_after_first_line(sample, fps)

                # channel selection
                if self.include_channels:
                    sample_np = np.array(sample, dtype=np.float32)[self.include_channels]
                else:
                    sample_np = np.array(sample, dtype=np.float32)

                self.raw_buf.append(sample_np)
                self.ts_buf.append(now)

                # push in small chunks
                if len(self.raw_buf) >= CHUNK_SAMPLES:
                    raw_chunk = np.stack([self.raw_buf.popleft() for _ in range(CHUNK_SAMPLES)], axis=0)
                    ts_chunk = [self.ts_buf.popleft() for _ in range(CHUNK_SAMPLES)]

                    # Update filter’s fs occasionally if our estimate drifts a lot
                    if abs(fps - self.fs_for_filter) / max(self.fs_for_filter, 1e-6) > 0.2:
                        self.fs_for_filter = fps
                        self.init_filter()  # re-init to keep behavior sane

                    # Filter
                    filt_chunk = self.iir.process(raw_chunk)

                    # LSL: both chunks are (num_samples, num_channels)
                    if self.raw_outlet:
                        self.raw_outlet.push_chunk(raw_chunk.tolist(), ts_chunk)
                    if self.filt_outlet:
                        self.filt_outlet.push_chunk(filt_chunk.tolist(), ts_chunk)

        except Exception as e:
            self.log.exception(f"Server error: {e}")
        finally:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.log.info("Serial closed. Exiting thread.")

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="CircuitPython Serial → LSL with IIR filtering")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baudrate")
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
        port=args.port,
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
