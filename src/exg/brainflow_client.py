# graph_lsl.py

import sys
import logging
import numpy as np
from PyQt5 import QtCore
from pylsl import StreamInlet, resolve_byprop, proc_ALL

from graph import BaseGraph

class LSLGraphPlotter(BaseGraph):
    def __init__(self, stream_name, stream_type="EXG", fps=50):
        # Initialize logging specific to this module
        self._init_logging()

        self.stream_name = stream_name
        self.stream_type = stream_type

        # Initialize LSL inlet
        self.inlet = self._resolve_stream(self.stream_name, self.stream_type)

        # Get stream info
        self.stream_info = self.inlet.info()

        self.num_channels = self.stream_info.channel_count()
        self.sampling_rate = self.stream_info.nominal_srate()

        logging.info(f"Number of channels: {self.num_channels}")
        logging.info(f"Sampling rate: {self.sampling_rate} Hz")

        # Initialize the base graph
        super().__init__(
            title=f'LSL {self.stream_name} Stream',
            window_size=5,
            sampling_rate=self.sampling_rate,
            num_channels=self.num_channels
        )

        # Initialize buffers
        self.exg_buffer = np.zeros((self.num_channels, self.buffer_size))
        self.timestamp_buffer = np.zeros(self.buffer_size)

        # Set up a timer to periodically fetch and plot data
        self.update_speed_ms = 1000 // fps  # 20 ms for ~50 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        logging.info("Starting Qt event loop.")
        self.app.exec_()

    def _init_logging(self):
        """
        Configures the logging settings.
        """
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for detailed logs
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging initialized for LSLGraphPlotter.")

    def _resolve_stream(self, name, type_):
        """
        Resolves an LSL stream by name and type.
        """
        logging.info(f"Resolving {type_} stream with name '{name}'...")
        streams = resolve_byprop('name', name)
        if not streams:
            logging.error(f"No stream found with name '{name}' and type '{type_}'. Exiting.")
            sys.exit(1)
        inlet = StreamInlet(streams[0], processing_flags=proc_ALL)
        logging.info(f"Connected to stream: {inlet.info().name()}")
        return inlet

    def update(self):
        """
        Fetches new data from the LSL streams and updates the plots.
        """
        try:
            # Fetch exg data
            chunk_exg, timestamp_exg = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
            if chunk_exg:
                chunk_exg = np.array(chunk_exg).T  # Shape: (channels, samples)
                num_samples_exg = chunk_exg.shape[1]
                self.exg_buffer = np.roll(self.exg_buffer, -num_samples_exg, axis=1)
                self.exg_buffer[:, -num_samples_exg:] = chunk_exg
                self.timestamp_buffer = np.roll(self.timestamp_buffer, -num_samples_exg)
                self.timestamp_buffer[-num_samples_exg:] = timestamp_exg[:num_samples_exg]

                # Update plots
                time_axis = np.linspace(-self.window_size, 0, self.buffer_size)

                for i in range(self.num_channels):
                    self.curves[i].setData(time_axis, self.exg_buffer[i])

        except Exception as e:
            logging.error(f"Error during update: {e}")

    def cleanup(self):
        """
        Overrides the cleanup method to perform any necessary cleanup.
        """
        # If there are resources to release, do it here
        pass

def main():
    # Configure logging (if not already configured in BaseGraph)
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for detailed logs
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        lsl_plotter = LSLGraphPlotter(stream_name='filtered_exg', stream_type='EXG')
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting.")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
    finally:
        logging.info("Program terminated.")

if __name__ == '__main__':
    main()
