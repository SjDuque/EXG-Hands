# realtime_prediction.py
import numpy as np
import pandas as pd
import time
from exg.ema import EMA
from exg.sma import SMA
from exg.iir import IIR
from tensorflow.keras.models import load_model
import pylsl
import json

# Path to the trained model
MODEL_DIR = "data/s_05_14_25/models/model_0"


class HandPredictionServer:
    def __init__(self,
                 raw_exg_stream: str,
                 model_dir: str,
                 fps: int = 30,
                 ):
        """
        Initializes the real-time predictor.

        :param recorder: Instance of EMARecorder for data collection.
        :param model_path: Path to the trained Keras model.
        """
        self.model = load_model(model_dir + "/model.keras")
        model_params = open(model_dir + "/model_params.json", 'r')
        model_params = json.load(model_params)
        self.finger_cols = model_params['finger_cols']
        self.x_cols = model_params['x_cols']
        self.exg_fs = model_params['exg_fs']
        self.ema_methods = model_params['ema_methods']
        self.ema_intervals = model_params['ema_intervals']
        self.sma_methods = model_params['sma_methods']
        self.sma_intervals = model_params['sma_intervals']
        self.iir_params = model_params['iir_params']
        
        self.mean = np.array(model_params['mean'])
        self.std = np.array(model_params['std'])
        
        self.num_fingers = 5
        self.prev_average_output = np.zeros(self.num_fingers)
        
        self._proc_flags = (
            pylsl.proc_clocksync |
            pylsl.proc_dejitter |
            pylsl.proc_monotonize
        )
        
        # Initialize the EXG stream
        # Resolve and open LSL inlet
        found = pylsl.resolve_byprop('name', raw_exg_stream, timeout=5)
        if not found:
            raise RuntimeError(f"No LSL stream named '{raw_exg_stream}' found.")
        self.exg_inlet = pylsl.StreamInlet(
            found[0], max_buflen=int(self.exg_fs*5), processing_flags=self._proc_flags
        )
        
        num_channels = self.exg_inlet.info().channel_count()
        fs = self.exg_inlet.info().nominal_srate()
        if fs != self.exg_fs:
            raise ValueError(f"Sampling frequency mismatch: {fs} != {self.exg_fs}")
        
        # Initialize the EMA processor
        self.ema_processor = None
        if self.ema_methods:
            self.ema_processor = EMA(
                num_channels=num_channels,
                window_intervals_ms=self.ema_intervals,
                methods=self.ema_methods,
                fs=self.exg_fs,
            )
        
        # Initialize the SMA processor
        self.sma_processor = None
        if self.sma_methods:
            self.sma_processor = SMA(
                num_channels=num_channels,
                window_intervals_ms=self.sma_intervals,
                methods=self.sma_methods,
                fs=self.exg_fs,
            )
        
        # Initialize the IIR filter
        self.iir_processor = IIR(
            num_channels=self.iir_params["num_channels"],
            fs=self.iir_params["fs"],
            lowpass_fs=self.iir_params["lowpass_fs"],
            highpass_fs=self.iir_params["highpass_fs"],
            notch_fs_list=self.iir_params["notch_fs_list"],
            filter_order=self.iir_params["filter_order"],
        )
        
        # Start the lsl stream
        self.info = pylsl.StreamInfo('finger_predictions', 'Markers', self.num_fingers, pylsl.IRREGULAR_RATE, 'float32', 'finger_predictions')
        self.outlet = pylsl.StreamOutlet(self.info)
        
        self.fps = fps
        self.update_interval = 1 / fps
        
    def span_to_alpha(self, span):
        return 2 / (span + 1)

    def preprocess_exg(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocesses the data by normalizing.

        :param data: Unprocessed data.
        :return: Normalized data.
        """
        # Normalize using pre-loaded mean and std
        normalized = (data - self.mean) / self.std
        return normalized

    def predict(self):
        """
        Continuously checks the buffer for new data and makes predictions.
        """
        data, _ = self.exg_inlet.pull_chunk(timeout=1.0)
        
        if data is None:
            return None
        # IIR processing
        iir_data = self.iir_processor.process(np.array(data))
        # EMA processing
        ema_data = None
        if self.ema_processor is not None:
            ema_data = self.ema_processor.process(iir_data)
            ema_data = self.ema_processor.results_to_df(ema_data)
        # SMA processing
        sma_data = None
        if self.sma_processor is not None:
            sma_data = self.sma_processor.process(iir_data)
            sma_data = self.sma_processor.results_to_df(sma_data)
        
        # Concatenate EMA and SMA data
        exg_data = pd.concat([ema_data, sma_data], axis=1)
        
        # Select the columns for prediction
        exg_data = exg_data[self.x_cols].values
        
        # Preprocess the data
        exg_data = self.preprocess_exg(exg_data)
        
        prediction = self.model.predict(exg_data, verbose=0)
        
        # Mean of the output
        average_output = np.mean(prediction, axis=0)
        # Binary prediction
        binary_prediction = average_output.round().astype(int)
        
        # Print the predictions
        print(f'Fingers: {self.finger_cols}')
        print(f"Predictions: {binary_prediction}")
        
        # If the number of outputs is 4, add the last finger twice
        if len(binary_prediction) == 4:
            # Add the last finger twice
            binary_prediction = np.append(binary_prediction, binary_prediction[-1])
        
        # Stream the predictions
        self.outlet.push_sample(binary_prediction)

    def run_loop(self):
        """
        Starts data collection and prediction.
        """
        last_time = time.perf_counter()
        while True:
            # Start prediction loop
            self.predict()
            print('...')
            # Sleep to maintain the desired FPS
            current_time = time.perf_counter()
            elapsed_time = current_time - last_time
            sleep_time = self.update_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

def main():

    # Initialize HandPredictionServer
    predictor = HandPredictionServer(
        raw_exg_stream="raw_exg",
        model_dir=MODEL_DIR,
        fps=3
    )

    # Run the real-time predictor
    predictor.run_loop()

if __name__ == "__main__":
    main()
