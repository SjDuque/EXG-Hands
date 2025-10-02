import logging
import sys
import time
import numpy as np
import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from pylsl import StreamInfo, StreamOutlet, local_clock
import serial.tools.list_ports
import threading

from iir import IIR

BOARD_ID = BoardIds.CYTON_BOARD  # Default to synthetic board for testing

class BrainFlowServer(threading.Thread):
    def __init__(self, fps:int=None, board_id:BoardIds=BoardIds.SYNTHETIC_BOARD, is_emg:bool=False, 
                 serial_port:str='', lsl_raw:bool=True, lsl_filtered:bool=True,
                 include_channels:list[int]=None):
        super().__init__()  # initialize the Thread
        self.running = False
        self._init_logging()
        
        # Automatically find the serial port if not provided
        if not serial_port and board_id is not BoardIds.SYNTHETIC_BOARD:
            serial_port = self._find_serial_port()
            if not serial_port:
                logging.error("No serial port found. Please specify a valid serial port.")
                sys.exit(1)
        logging.info(f"Using serial port: {serial_port}")
        
        # Set board parameters
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = board_id
        self.is_emg = is_emg

        # Initialize the board
        try:
            self.board_shim = BoardShim(self.board_id, self.params)
            logging.info(f"Initializing BoardShim with board_id: {self.board_id}")
            self.board_shim.prepare_session()
            logging.info("Board session prepared successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to prepare session: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error during board initialization: {e}")
            sys.exit(1)

        # Retrieve EXG channels and sampling rate
        try:
            if include_channels:
                self.exg_channels = include_channels
            else:
                self.exg_channels = BoardShim.get_exg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            logging.info(f"EXG Channels: {self.exg_channels}")
            logging.info(f"Sampling Rate: {self.sampling_rate} Hz")
        except BrainFlowError as e:
            logging.error(f"Failed to retrieve board information: {e}")
            self.release_board()
            sys.exit(1)
            
                
        if fps is None:
            fps = self.sampling_rate
        
        self.fps = fps

        # Configure board if EMG is enabled
        if self.is_emg and self.board_id in (BoardIds.GANGLION_BOARD, BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD):
            self._configure_openbci_emg()

        # Start streaming
        try:
            stream_buffer_size = int(self.sampling_rate * 5)  # 5 seconds of data
            self.board_shim.start_stream(stream_buffer_size, '')
            logging.info("Board streaming started successfully.")
        except BrainFlowError as e:
            logging.error(f"Failed to start board streaming: {e}")
            self.release_board()
            sys.exit(1)

        # Initialize LSL streams
        self.lsl_raw = lsl_raw
        self.lsl_filtered = lsl_filtered
        self._init_lsl_streams()

        # Compute time difference for LSL timestamps
        self.time_diff = local_clock() - datetime.datetime.now().timestamp()

        # Initialize the IIR filter:
        #   - lowpass_fs is set to half the sampling rate (i.e. the upper cutoff)
        #   - highpass_fs is set to 10.0 Hz (i.e. the lower cutoff)
        #   - notch_fs_list is set to remove 60 Hz interference.
        self.iir_filter = IIR(
            num_channels=len(self.exg_channels),
            fs=self.sampling_rate,
            lowpass_fs=self.sampling_rate / 2,
            highpass_fs=10.0,
            notch_fs_list=[50, 60],
            filter_order=4
        )

    def _init_logging(self):
        """
        Configures the logging settings.
        """
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.info("Logging initialized for BrainFlowServer.")

    def _init_lsl_streams(self):
        """
        Initializes LSL streams for raw and filtered data.
        """
        if self.lsl_raw:
            self.raw_lsl_info = StreamInfo(
                name="raw_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id="BrainFlowLSLStream"
            )
            self.raw_lsl_outlet = StreamOutlet(self.raw_lsl_info)
            logging.info("Raw LSL initialized.")

        if self.lsl_filtered:
            self.filtered_lsl_info = StreamInfo(
                name="filtered_exg",
                type="EXG",
                channel_count=len(self.exg_channels),
                nominal_srate=self.sampling_rate,
                channel_format="float32",
                source_id="BrainFlowLSLStream"
            )
            self.filtered_lsl_outlet = StreamOutlet(self.filtered_lsl_info)
            logging.info("Filtered LSL initialized.")
            
    def _find_serial_port(self):
        """
        Automatically detect the serial port for the OpenBCI board.
        Returns the port name as a string, or raises an exception if not found.
        """
        for port in serial.tools.list_ports.comports():
            # Check for specific keywords or VID/PID
            if 'FT231X USB UART' in port.description:
                return port.device
        logging.warning("OpenBCI board not found. Try setting your own serial port in brainflow_server.py.")
        return ''
    
    def _configure_openbci_emg(self):
        try:
            # Configuration parameters
            POWER_DOWN = 0  # EMG is 0 = Normal operation
            GAIN_SET = 6    # EMG is 6 = Gain 24
            INPUT_TYPE_SET = 0  # EMG is 0 = Normal electrode input
            BIAS_SET = 0    # EMG is 1
            SRB2_SET = 0    # EMG is 0
            SRB1_SET = 0    # EMG is 0
            
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
                logging.info("Configured settings for EMG.")
        except BrainFlowError as e:
            logging.error(f"Failed to configure board for EMG: {e}")
            self.release_board()
            sys.exit(1)

    def update_lsl_streams(self, raw_exg:np.ndarray, filtered_exg:np.ndarray, timestamp_data:np.ndarray):
        """
        Pushes the latest data chunks to the LSL streams.
        
        :param num_samples: Number of new samples to push
        """
        if self.lsl_raw:
            self.raw_lsl_outlet.push_chunk(raw_exg.tolist(), timestamp_data.tolist())

        if self.lsl_filtered:
            self.filtered_lsl_outlet.push_chunk(filtered_exg.tolist(), timestamp_data.tolist())

    def update(self):
        """
        Fetches new data from the board, processes it using the IIR filter, 
        and streams via LSL.
        """
        try:
            # Fetch the latest samples from the board
            new_data = self.board_shim.get_board_data().T # Shape: (num_samples, num_channels)
            if new_data is None or new_data.size == 0:
                return

            raw_exg = new_data[:, self.exg_channels] # Shape: (num_samples, num_channels)
            timestamp_data = new_data[:, self.timestamp_channel] + self.time_diff # Shape: (num_samples, 1)

            filtered_exg = self.iir_filter.process(raw_exg) # Shape: (num_samples, num_channels)
            # Update LSL streams with new data
            self.update_lsl_streams(raw_exg, filtered_exg, timestamp_data)

        except BrainFlowError as e:
            logging.error(f"BrainFlowError in update function: {e}")
            self.release_board()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error in update function: {e}")
            self.release_board()
            sys.exit(1)

    def run(self):
        """
        Threaded data acquisition loop.
        """
        self.running = True
        logging.info("Threaded BrainFlowServer started.")
        try:
            while self.running:
                self.update()
                time.sleep(1 / self.fps)
        except Exception as e:
            logging.error(f"Exception in BrainFlowServer thread: {e}")
            self.cleanup()
            
    def stop(self):
        """
        Safely stop the server.
        """
        logging.info("Stopping BrainFlowServer thread.")
        self.running = False
        self.cleanup()

    def release_board(self):
        """
        Releases board resources.
        """
        try:
            if self.board_shim.is_prepared():
                logging.info('Stopping and releasing board session.')
                self.board_shim.stop_stream()
                self.board_shim.release_session()
        except Exception as e:
            logging.error(f"Error releasing board: {e}")

    def cleanup(self):
        """
        Releases the board and performs cleanup.
        """
        self.release_board()
        logging.info("Cleanup complete. Exiting.")

def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Enable BrainFlow's internal logger if needed
    BoardShim.enable_dev_board_logger()
    
    # Create and start the BrainFlowServer
    logging.info("Starting BrainFlowServer...")
    server = BrainFlowServer(board_id=BOARD_ID, is_emg=True)
    server.start()
    # Wait for the server to start
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping BrainFlowServer...")
        server.stop()
        server.join()

if __name__ == '__main__':
    main()