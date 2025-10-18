import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
from pylsl import StreamInfo, StreamOutlet
from serial.tools import list_ports

from filtering.iir import IIR


class BaseExgServer(threading.Thread, ABC):
    """
    Shared functionality for threaded EXG acquisition servers.

    Child classes implement acquisition details while inheriting filtering,
    LSL streaming, and lifecycle management.
    """

    def __init__(
        self,
        *,
        name: str,
        fps: float | None = None,
        lsl_raw: bool = True,
        lsl_filtered: bool = True,
        lowpass_fs: float | None = None,
        highpass_fs: float | None = None,
        notch_fs_list: list[float] | None = None,
        filter_order: int = 4,
        raw_stream_name: str = "raw_exg",
        filtered_stream_name: str = "filtered_exg",
        source_id: str | None = None,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self.log = logging.getLogger(name)
        self.running = False
        self.fps = fps
        self._configure_logging()

        self.lsl_raw = lsl_raw
        self.lsl_filtered = lsl_filtered

        self._lowpass = lowpass_fs
        self._highpass = highpass_fs
        self._notch_list = notch_fs_list
        self._filter_order = filter_order
        
        self.raw_stream_name = raw_stream_name
        self.filtered_stream_name = filtered_stream_name
        self.lsl_source_id = source_id or self.__class__.__name__

        self.sample_rate: float | None = None

        self.raw_outlet: StreamOutlet | None = None
        self.filtered_outlet: StreamOutlet | None = None
        self.iir: IIR | None = None
        self._closed = False
        
    # ------------------------------------------------------------------
    # Properties
    @property
    def is_running(self) -> bool:
        return self.running

    @property
    def lowpass(self) -> float | None:
        return self._lowpass
    
    @property
    def highpass(self) -> float | None:
        return self._highpass
    
    @property
    def notch_list(self) -> list[float] | None:
        return self._notch_list
    

    # ------------------------------------------------------------------
    # Lifecycle helpers

    def start(self) -> None:
        self.log.info("Starting %s thread.", self.__class__.__name__)
        super().start()

    def run(self) -> None:
        delay = 0.0
        if self.fps and self.fps > 0:
            delay = 1.0 / self.fps

        self.running = True
        self.log.info("Acquisition loop entered.")
        try:
            while self.running:
                result = self._acquire_samples()
                if result is not None:
                    samples, timestamps = result
                    if samples.size:
                        self._process_chunk(samples, timestamps)
                if delay:
                    time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Unhandled exception in acquisition loop: %s", exc)
            self.running = False
        finally:
            self.close()

    def stop(self) -> None:
        self.log.info("Stop requested.")
        self.running = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.log.info("BaseExgServer cleanup complete.")

    # ------------------------------------------------------------------
    # Configuration helpers
    
    def _configure_logging(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            )

    def _configure_processing(self, sample_rate: float, channel_count: int) -> None:
        """
        Configure internal processing once the sampling rate and number of input
        channels are known. This should be called before the first chunk is processed.
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if channel_count <= 0:
            raise ValueError("Input channel count must be positive.")

        self.sample_rate = sample_rate
        self.channel_count = channel_count

        self._init_lsl_streams()
        self.update_filter(
            lowpass_fs=self._lowpass,
            highpass_fs=self._highpass,
            notches_fs=self._notch_list,
            filter_order=self._filter_order
        )
        
    # ------------------------------------------------------------------
    # Filter management
    def update_filter(
            self,
            lowpass_fs: float | None=None,
            highpass_fs: float | None=None,
            notches_fs: list[float] | None=None,
            filter_order: int=None,
    ) -> None:
        """
        Rebuild the IIR filter, sample rate and channel count must already be set.
        If any parameter is None, the existing value is retained.
        """
        if self.sample_rate is None or self.channel_count is None:
            raise RuntimeError("Sample rate or channel count not set. Call _configure_processing() first.")
        nyquist: float = self.sample_rate / 2.0
        
        # If parameters are None, retain existing values
        self._lowpass = lowpass_fs or self._lowpass
        self._highpass = highpass_fs or self._highpass
        self._notch_list = notches_fs or self._notch_list or []
        self._filter_order = filter_order or self._filter_order
        
        # Validate cutoff frequencies
        if self._lowpass is None or self._lowpass <= 0 or self._lowpass > nyquist:
            self._lowpass = nyquist
            
        if self._highpass is None or self._highpass < 0 or self._highpass >= nyquist:
            self._highpass = 10.0
            
        if self._highpass >= self._lowpass:
            self._lowpass = nyquist
            self._highpass = 10.0
    
        # Validate notch frequencies
        self._notch_list = [f for f in self._notch_list if 0 < f < nyquist]
        
        # Validate filter order
        if self._filter_order is None or self._filter_order % 2 != 0 or self._filter_order <= 0:
            self._filter_order = 4
    
        # Build the IIR filter
        self.iir = IIR(
            num_channels=self.channel_count,
            fs=self.sample_rate,
            lowpass_fs=self._lowpass,
            highpass_fs=self._highpass,
            notch_fs_list=self._notch_list,
            filter_order=self._filter_order,
        )
        self.log.info(f"Filter Initialized: lowpass={self._lowpass}Hz, highpass={self._highpass}Hz, notches={self._notch_list}, order={self._filter_order}")
        
    def _apply_filter(self, samples: np.ndarray) -> np.ndarray:
        if self.iir is None:
            raise RuntimeError("Filter not initialized. Call configure_processing() first.")
        return self.iir.process(samples)
    
    # ------------------------------------------------------------------
    # LSL management
    def _init_lsl_stream(self, name: str) -> StreamOutlet:
        info = StreamInfo(
            name=name,
            type="EXG",
            channel_count=self.channel_count,
            nominal_srate=self.sample_rate,
            channel_format="float32",
            source_id=self.lsl_source_id,
        )
        ret = StreamOutlet(info)
        self.log.info("LSL %s outlet initialized.", name)
        return ret

    def _init_lsl_streams(self) -> None:
        if self.lsl_raw:
            self.raw_outlet = self._init_lsl_stream(name=self.raw_stream_name)
        if self.lsl_filtered:
            self.filtered_outlet = self._init_lsl_stream(name=self.filtered_stream_name)

    def _push_to_lsl(self, timestamps: list[float], raw_chunk: np.ndarray, filtered_chunk: np.ndarray | None = None,
    ) -> None:
        if self.raw_outlet:
            self.raw_outlet.push_chunk(raw_chunk.tolist(), list(timestamps))
        if filtered_chunk is not None and self.filtered_outlet:
            self.filtered_outlet.push_chunk(filtered_chunk.tolist(), list(timestamps))
            
    
    def _process_chunk(
        self, samples: np.ndarray, timestamps: list[float]
    ) -> np.ndarray:
        """
        Apply optional channel selection, filter the chunk, and push to LSL.
        Returns the filtered chunk for additional downstream handling.
        """
        if samples.size == 0:
            return samples
        filt = self._apply_filter(samples)
        self._push_to_lsl(raw_chunk=samples, filtered_chunk=filt, timestamps=timestamps)
        return filt

    # ------------------------------------------------------------------
    # Abstract methods
    @abstractmethod
    def _acquire_samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the next chunk as (samples, timestamps), or None when no data is available."""


def resolve_serial_port(port: str | None = None, desc: str | None = None, manu: str | None = None) -> str | None:
    """
    Resolve a serial port by explicit name, description substring, or manufacturer substring.
    Parameters:
        port: If given and non-empty, use this port directly.
        desc: If port is not given, search for a port whose description contains this substring.
        manu: If port is not given, search for a port whose manufacturer contains this substring.
    Returns:
        The resolved port name, or None if no matching port is found.
    """
    # If port is given and not empty, use it directly.
    if port:
        return port
    # If neither description nor manufacturer is given, cannot resolve.
    if not desc and not manu:
        return ""
    # Search available ports for a match.
    ports = list_ports.comports()
    for port in ports:
        if desc and desc in port.description:
            return port.device
        if manu and port.manufacturer and manu in port.manufacturer:
            return port.device
    return ""