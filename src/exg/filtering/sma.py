import numpy as np
import pandas as pd

class SMA:
    """
    _summary_: This class implements a simple moving average (SMA) processor with multiple window sizes.
    """
    def __init__(self, num_channels:int, fs:float, window_intervals_ms:list[int]=None, window_sizes:list[int]=None,
                 methods:list[str] = ['mean', 'mean_square', 'root_mean_square', 'variance', 'standard_deviation']
                 ):
        """
        Initialize the simple moving average.
        NOTE: ONLY WORKS FOR FS <= 1000 Hz DUE TO MS PRECISION

        Args:
            num_channels (int): Number of signal channels.
            fs (float): Sampling frequency in Hz.
            window_intervals_ms (list): List of window intervals in ms. (Choose either window_intervals or window_sizes.)
            window_sizes (list): List of window sizes in number of sample with millisecond precision. (Choose either window_intervals or window_sizes.)
            methods (list): List of processing methods to compute. Default: ALL
        """
        # Initialize window sizes
        if window_intervals_ms is None and window_sizes is None:
            raise ValueError("Either window_intervals or window_sizes must be provided.")
        elif window_intervals_ms is not None and window_sizes is not None:
            raise ValueError("Only one of window_intervals or window_sizes can be provided.")
        elif window_intervals_ms is not None:
            window_sizes = [int(round(interval/1000 * fs)) for interval in window_intervals_ms]

        # Set window_sizes to be unique and a numpy array
        self.window_sizes = np.unique(window_sizes)
        
        # Verify all window_sizes are above 0
        if np.any(self.window_sizes <= 0):
            raise ValueError("All window sizes must be greater than 0.")
        
        # Set window_intervals
        self.window_intervals_ms = ((self.window_sizes / fs) * 1000).round().astype(np.int32)
        
        # Initialize processing methods
        self.methods = set(methods)

        # Modify the methods to include the requisite methods
        if "standard_deviation" in self.methods:
            self.methods.add("variance")

        if "variance" in self.methods:
            self.methods.add("mean_square")

        if "root_mean_square" in self.methods:
            self.methods.add("mean_square")

        if "mean_square" in self.methods:
            self.methods.add("mean")
            
        if "mean" not in self.methods:
            raise ValueError("Mean must be included in the methods.")

        # Initialize attributes
        self.num_channels    = num_channels
        self.num_windows     = len(self.window_sizes)
        self.max_window_size = self.window_sizes[-1]

        # Initialize rolling buffers
        self.idx            = 0
        self.prev_vals      = np.zeros((self.max_window_size, num_channels), dtype=np.float64)
        self.sum        = np.zeros((self.num_windows, num_channels), dtype=np.float64)
        if "mean_square" in self.methods:
            self.sum_square = np.zeros((self.num_windows, num_channels), dtype=np.float64)

    def process(self, new_vals:np.ndarray | list[list[float]]) -> dict[str, dict[int, np.ndarray]]:
        """
        Update the rolling processor with new values for all channels and compute selected methods.

        Args:
            new_vals (ndarray): New signal values for all channels. Shape: (num_samples, num_channels).

        Returns:
            dict: Metrics for each method
                  Each metric is a NumPy array of shape (num_samples, num_windows, num_channels).
        """
        # Rectify the signal
        new_vals = np.abs(new_vals)

        # Check that new_vals is a 2D array
        if new_vals.ndim != 2:
            raise ValueError(f"new_vals must be a 2D array of shape (num_samples, {self.num_channels}). Got shape: {new_vals.shape}")

        # Get the number of samples and channels        
        num_samples, num_channels = new_vals.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Number of channels ({num_channels}) does not match the expected value ({self.num_channels}).")

        # Initialize results for mean and mean_square
        results = {}
        new_shape = (num_samples, self.num_windows, num_channels)
        results['mean'] = np.zeros(new_shape, dtype=np.float64)
        has_mean_square = "mean_square" in self.methods
        if has_mean_square:
            results['mean_square'] = np.zeros(new_shape, dtype=np.float64)

        # Iterate over each sample
        denom = self.window_sizes.reshape(-1, 1) # Denominator for computing the mean
        for s in range(num_samples):
            old_sample = self.prev_vals[self.idx - self.window_sizes]
            new_sample = new_vals[s]

            # Update the sum and mean
            self.sum += new_sample - old_sample
            self.sum = np.clip(self.sum, 0, None)
            results["mean"][s, :, :] = self.sum / denom

            # Update the sum of squares and mean square
            if has_mean_square:
                self.sum_square += new_sample**2 - old_sample**2
                self.sum_square = np.clip(self.sum_square, 0, None)
                results["mean_square"][s, :, :] = self.sum_square / denom

            # Update the rolling buffer
            self.prev_vals[self.idx, :] = new_sample
            self.idx                    = (self.idx + 1) % self.max_window_size

        # Compute additional metrics
        if "root_mean_square" in self.methods:
            # Compute the root mean squared value in the current window
            results["root_mean_square"] = np.sqrt(results['mean_square'])

        if "variance" in self.methods:
            # Compute the variance in the current window
            variance = results["mean_square"] - results["mean"]**2
            variance = np.clip(variance, 0, None)
            results["variance"] = variance

        if "standard_deviation" in self.methods:
            # Compute the standard deviation in the current window
            results["standard_deviation"] = np.sqrt(results["variance"])

        return results

    def results_to_df(self, results: dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Convert the results to a Pandas DataFrame
            results[method] has shape = (num_samples, num_windows, num_channels)
        """
        # Prepare a dictionary for DataFrame columns
        data_dict = {}
        methods = sorted(self.methods)
        
        for method in methods:
            res_array = results[method]  # shape: (num_samples, num_windows, num_channels)
            
            if res_array.ndim != 3:
                raise ValueError(f"Results for method '{method}' must be a 3D array. Got shape: {res_array.shape}")
            
            _, num_windows, num_channels = res_array.shape
            
            # Check if the number of windows and channels match the class values
            if num_windows != self.num_windows:
                raise ValueError(f"Number of windows ({num_windows}) does not match the expected value ({self.num_windows}).")
            if num_channels != self.num_channels:
                raise ValueError(f"Number of channels ({num_channels}) does not match the expected value ({self.num_channels}).")
            
            # Iterate over each window
            for w in range(self.num_windows):
                interval = self.window_intervals_ms[w]
                
                # Iterate over each channel
                for c in range(self.num_channels):
                    col_name = f"ch_{c+1}_sma_{method}_{interval}ms"
                    # Flatten across `num_samples` for this window+channel
                    data_dict[col_name] = res_array[:, w, c]
        
        # Now create a DataFrame from the dictionary
        df = pd.DataFrame(data_dict)
        return df

    def reset(self):
        """
        Reset the rolling processor.
        """
        self.idx = 0
        self.prev_vals.fill(0)
        self.sum.fill(0)
        if "mean_square" in self.methods:
            self.sum_square.fill(0)

        print("Processor reset.")

def main():
    # Define parameters
    window_intervals_ms = [20, 50, 100]  # Rolling window sizes in milliseconds
    num_channels = 8  # Number of EMG/EEG channels
    fs = 250.0  # Sampling frequency in Hz

    methods = ["mean", "mean_square", "root_mean_square", "variance", "standard_deviation"]

    # Initialize the rolling processor
    processor = SMA(
        num_channels=num_channels,
        fs=fs,
        methods=methods,
        window_intervals_ms=window_intervals_ms
    )

    # Generate random data
    num_samples = 100000
    data = np.random.randn(num_samples, num_channels)
    # Process the data
    results = processor.process(data)
    # Mean values in the first window
    print('Mean:', results["mean"][:, 0].mean())
    # Standard deviation values in the second window
    print('STD:', results["standard_deviation"][:, 0].mean())

    print("Data acquisition and processing completed.")

if __name__ == "__main__":
    main()
