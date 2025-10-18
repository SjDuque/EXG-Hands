import datetime
import sys
from collections.abc import Sequence

import numpy as np
from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowError,
    BrainFlowInputParams,
)
from pylsl import local_clock

from .base_exg_server import BaseExgServer, resolve_serial_port


class BrainFlowServer(BaseExgServer):
    def __init__(
        self,
        fps: int | None = None,
        board_id: BoardIds = BoardIds.SYNTHETIC_BOARD,
        is_bipolar: bool = True,
        serial_port: str = "",
        lsl_raw: bool = True,
        lsl_filtered: bool = True,
        lowpass_fs: float | None = None,
        highpass_fs: float | None = None,
        notch_fs_list: list[float] | None = None,
    ) -> None:
        if highpass_fs is None:
            highpass_fs = 10.0
        if notch_fs_list is None:
            notch_fs_list = [50.0, 60.0]
        super().__init__(
            name="BrainFlowServer",
            fps=fps,
            lsl_raw=lsl_raw,
            lsl_filtered=lsl_filtered,
            lowpass_fs=lowpass_fs,
            highpass_fs=highpass_fs,
            notch_fs_list=notch_fs_list,
            raw_stream_name="raw_exg",
            filtered_stream_name="filtered_exg",
            source_id="BrainFlowLSLStream",
            daemon=False,
        )

        serial_port = resolve_serial_port(
            serial_port,
            desc="FT231X USB UART"
        )
        self.log.info("Using serial port: %s", serial_port or "<none>")

        # Set board parameters
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        
        # Initialize the board
        try:
            self.board_shim: BoardShim = BoardShim(self.board_id, self.params)
            self.log.info("Initializing BoardShim with board_id: %s", self.board_id)
            self.board_shim.prepare_session()
            self.log.info("Board session prepared successfully.")
        except BrainFlowError as exc:
            self.log.error("Failed to prepare session: %s", exc)
            sys.exit(1)
        except Exception as exc:
            self.log.error("Unexpected error during board initialization: %s", exc)
            sys.exit(1)

        # Retrieve EXG channels and sampling rate
        try:
            self.exg_channels = BoardShim.get_exg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            self.log.info("EXG Channels: %s", self.exg_channels)
            self.log.info("Sampling Rate: %s Hz", self.sampling_rate)
        except BrainFlowError as exc:
            self.log.error("Failed to retrieve board information: %s", exc)
            self.release_board()
            sys.exit(1)

        if self.fps is None:
            self.fps = self.sampling_rate

        # Configure board if EMG is enabled
        if self.board_id in (
            BoardIds.GANGLION_BOARD,
            BoardIds.CYTON_BOARD,
            BoardIds.CYTON_DAISY_BOARD,
        ):
            self._configure_openbci(is_bipolar=is_bipolar)

        # Start streaming
        try:
            stream_buffer_size = int(self.sampling_rate * 5)  # 5 seconds of data
            self.board_shim.start_stream(stream_buffer_size, "")
            self.log.info("Board streaming started successfully.")
        except BrainFlowError as exc:
            self.log.error("Failed to start board streaming: %s", exc)
            self.release_board()
            sys.exit(1)

        # Initialize shared processing
        self._configure_processing(
            sample_rate=self.sampling_rate,
            channel_count=len(self.exg_channels),
        )

        # Compute time difference for LSL timestamps
        self.time_diff = local_clock() - datetime.datetime.now().timestamp()

    def _configure_openbci(self, is_bipolar: bool) -> None:
        try:
            # Configuration parameters
            POWER_DOWN = 0  #  0 = Normal operation
            GAIN_SET = 6    #  6 = Gain 24
            INPUT_TYPE_SET = 0  # 0 = Normal electrode input
            BIAS_SET = 1    # 1 = Enable Bias Electrode
            
            if is_bipolar:
                SRB2_SET = 0    # 0 = Bipolar mode
            else:
                SRB2_SET = 1    # 1 = Unipolar mode
            SRB1_SET = 0    # 0 = Disconnect
            
            # Channel names: 1 2 3 4 5 6 7 8 Q W E R T Y U I
            channel_names = ['1', '2', '3', '4', '5', '6', '7', '8', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I']
            # Ensure each value is a single digit
            def valid_value(value):
                return len(str(value)) == 1

            if not all(map(valid_value, [POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET])):
                raise ValueError("Invalid value for config settings (All must be a single character).")

            # Set config settings for each channel
            config_list = [f"x{channel_names[channel-1]}{POWER_DOWN}{GAIN_SET}{INPUT_TYPE_SET}{BIAS_SET}{SRB2_SET}{SRB1_SET}X" 
                            for channel in self.exg_channels]
            config = ''.join(config_list)
            if config:
                self.board_shim.config_board(config)
                self.log.info("Configured settings for EMG.")
        except BrainFlowError as exc:
            self.log.error("Failed to configure board for EMG: %s", exc)
            self.release_board()
            sys.exit(1)

    def _acquire_samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Fetch new data from the board and return (samples, timestamps)."""
        try:
            # Fetch the latest samples from the board
            new_data = self.board_shim.get_board_data().T # Shape: (num_samples, num_channels)
            if new_data is None or new_data.size == 0:
                return None

            raw_exg = new_data[:, self.exg_channels]  # Shape: (num_samples, num_channels)
            timestamp_data = new_data[:, self.timestamp_channel] + self.time_diff  # Shape: (num_samples, 1)

            return raw_exg, timestamp_data
        except BrainFlowError as exc:
            self.log.error("BrainFlowError in update function: %s", exc)
            self.release_board()
            sys.exit(1)
        except Exception as exc:
            self.log.error("Unexpected error in update function: %s", exc)
            self.release_board()
            sys.exit(1)

    def close(self) -> None:
        """
        Releases board resources.
        """
        try:
            if self.board_shim.is_prepared():
                self.log.info('Stopping and releasing board session.')
                self.board_shim.stop_stream()
                self.board_shim.release_session()
        except Exception as exc:
            self.log.error("Error releasing board: %s", exc)
        
        self.log.info("Cleanup complete. Exiting.")
        super().close()
