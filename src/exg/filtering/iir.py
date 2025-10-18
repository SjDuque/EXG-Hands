import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

class IIR:
    def __init__(self, num_channels: int, fs: float, lowpass_fs: float, highpass_fs: float, 
                 notch_fs_list: list[float], filter_order: int = 4):
        self.num_channels = num_channels
        self.fs = fs
        self.sos_combined = self._combine_filters(lowpass_fs, highpass_fs, notch_fs_list, filter_order)
        self.num_sections = self.sos_combined.shape[0]
        # Initialize zi for each channel and section
        self.zi = np.zeros((num_channels, self.num_sections, 2))
    
    def _combine_filters(self, lowpass_fs: float, highpass_fs: float, notch_fs_list: list[float], 
                         filter_order: int) -> np.ndarray:
        # Design bandpass filter
        sos_bp = self.design_bandpass_filter(self.fs, lowpass_fs, highpass_fs, order=filter_order)
        
        # Design all notch filters
        sos_notch_list = [self.design_bandstop_filter(self.fs, nf-4, nf+4, order=2) 
                          for nf in notch_fs_list]
        # sos_notch_list = [self.design_notch_filter(self.fs, nf, quality_factor=7)
        #                     for nf in notch_fs_list]
        
        # Combine all SOS arrays: bandpass first, then all notch filters
        sos_combined = [sos_bp] + sos_notch_list
        
        sos_combined = np.vstack(sos_combined)
        
        return sos_combined
    
    @staticmethod
    def design_bandpass_filter(fs: float, lowpass_fs: float, highpass_fs: float,
                               order: int = 4, eps: float = 1e-8) -> np.ndarray:
        
        if lowpass_fs <= highpass_fs:
            raise ValueError(f"Low pass frequency must be more than high pass frequency. {lowpass_fs} <= {highpass_fs}")
        
        nyquist = 0.5 * fs
        Wn = [highpass_fs / nyquist, lowpass_fs / nyquist]

        # Ensure cutoff frequencies are within valid range
        Wn[0] = max(Wn[0], eps)  # Ensure low cutoff is at least 0.00...1% of Nyquist
        Wn[1] = min(Wn[1], 1 - eps)  # Ensure high cutoff is at most 99.99...9% of Nyquist

        sos = butter(order, Wn=Wn, btype='bandpass', analog=False, output='sos')
        return sos
    
    @staticmethod
    def design_bandstop_filter(fs: float, lowstop_fs: float, highstop_fs: float,
                               order: int = 4, eps: float = 1e-8) -> np.ndarray:
        
        if lowstop_fs >= highstop_fs:
            raise ValueError(f"High stop frequency must be greater than low stop frequency. {highstop_fs} >= {lowstop_fs}")
        
        nyquist = 0.5 * fs
        Wn = [lowstop_fs / nyquist, highstop_fs / nyquist]

        # Ensure cutoff frequencies are within valid range
        Wn[0] = max(Wn[0], eps)
        Wn[1] = min(Wn[1], 1 - eps)
        sos = butter(order, Wn=Wn, btype='bandstop', analog=False, output='sos')
        return sos

    @staticmethod
    def design_notch_filter(fs: float, notch_fs: float, quality_factor: float = 50, eps: float = 1e-8) -> np.ndarray:
        nyquist = 0.5 * fs
        w0 = notch_fs / nyquist
        w0 = max(w0, eps)  # Ensure w0 is at least 0.00...1% of Nyquist
        w0 = min(w0, 1 - eps)  # Ensure w0 is at most 99.99...9% of Nyquist
        b, a = iirnotch(w0, quality_factor)
        sos = tf2sos(b, a)
        return sos

    def process_inplace(self, data: np.ndarray) -> np.ndarray:
        """
        Apply combined bandpass and notch filters to multi-channel data in real-time.

        Args:
            data (ndarray): New data batch (num_samples, num_channels).

        """
        num_channels = data.shape[1]
        
        for ch in range(num_channels):
            # Apply the combined SOS filter to each channel with its own state
            data[:, ch], self.zi[ch] = sosfilt(self.sos_combined, data[:, ch], zi=self.zi[ch])
            
        return data
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply combined bandpass and notch filters to multi-channel data in real-time.

        Args:
            data (ndarray): New data batch (num_samples, num_channels).

        """
        num_channels = data.shape[1]
        
        filtered_data = np.zeros_like(data)
        
        for ch in range(num_channels):
            # Apply the combined SOS filter to each channel with its own state
            filtered_data[:, ch], self.zi[ch] = sosfilt(self.sos_combined, data[:, ch], zi=self.zi[ch])
            
        return filtered_data
    
    def reset(self):
        self.zi = np.zeros((self.num_channels, self.num_sections, 2))

if __name__ == "__main__":
    # Test the IIR filter
    num_channels = 2
    fs = 1000
    lowpass_fs = 100
    highpass_fs = 1
    notch_fs_list = [50, 100]
    filter_order = 4
    iir = IIR(num_channels, fs, lowpass_fs, highpass_fs, notch_fs_list, filter_order)
    
    # Generate a combined signal at notch frequencies + noise
    num_samples = 1000
    time = np.arange(num_samples) / fs
    signal = np.zeros((num_samples, num_channels))
    signal_fs_list = [50]
    for ch in range(num_channels):
        for fs in signal_fs_list:
            signal[:, ch] += np.sin(2 * np.pi * fs * time)
    # noise = np.random.randn(num_samples, num_channels)
    combined_signal = signal 
    
    # Mean of the combined signal
    print('Mean:', np.abs(combined_signal).mean())
    
    # Apply the IIR filter
    filtered_signal = iir.apply_inplace(combined_signal)
    
    # Print the mean of the filtered signal
    print('Mean:', np.abs(filtered_signal).mean())
    
    
            