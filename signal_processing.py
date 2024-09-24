# signal_processing.py

import numpy as np
from scipy.signal import butter, lfilter

class SignalProcessor:
    def __init__(self, sampling_rate, lowcut, highcut):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
    
    def filter_signals(self, signals):
        filtered_signals = [self._bandpass_filter(signal) for signal in signals]
        return filtered_signals
    
    def _bandpass_filter(self, data, order=5):
        nyq = 0.5 * self.sampling_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    
    def perform_beamforming(self, mic_positions, mic_signals, heatmap_type):
        # Define beamforming grid
        grid_resolution = 0.1  # meters
        x_grid = np.arange(-3, 3, grid_resolution)
        y_grid = np.arange(-3, 3, grid_resolution)
        z_grid = np.arange(1, 5, grid_resolution)  # Depth from 1 to 5 meters
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        
        # Perform beamforming
        intensity_map = self._delay_and_sum_beamforming(
            mic_positions,
            mic_signals,
            grid_points,
            heatmap_type
        )
        
        # Reshape intensity map for visualization
        intensity_map_reshaped = intensity_map.reshape(X.shape)
        return intensity_map_reshaped
    
    def _delay_and_sum_beamforming(self, mic_positions, mic_signals, grid_points, heatmap_type):
        num_points = grid_points.shape[0]
        num_mics = mic_positions.shape[0]
        intensity_map = np.zeros(num_points)
        
        # Frequency range for analysis
        num_samples = len(mic_signals[0])
        freq_bins_full = np.fft.fftfreq(num_samples, d=1/self.sampling_rate)
        freq_indices = np.where((freq_bins_full >= self.lowcut) & (freq_bins_full <= self.highcut))[0]
        freq_bins = freq_bins_full[freq_indices]
        
        mic_signals_fft = np.fft.fft(mic_signals, axis=1)[:, freq_indices]
        
        for idx, gp in enumerate(grid_points):
            # Compute distances and delays for all microphones
            distances = np.linalg.norm(mic_positions - gp, axis=1)  # Shape: (num_mics,)
            delays = distances / 343.0  # Speed of sound in m/s
            
            # Compute phase shifts for all microphones and frequencies
            # Shape of phase_shifts: (num_mics, num_freqs)
            phase_shifts = -2 * np.pi * freq_bins[None, :] * delays[:, None]
            
            # Compute steering vector
            # Shape of steering_vector: (num_mics, num_freqs)
            steering_vector = np.exp(1j * phase_shifts)
            
            # Beamforming: Sum the microphone signals after applying the steering vector
            # Multiply mic_signals_fft (num_mics, num_freqs) element-wise with steering_vector
            # Sum over microphones (axis=0) to get summed_signal of shape (num_freqs,)
            summed_signal = np.sum(mic_signals_fft * steering_vector, axis=0)
            
            # Calculate intensity based on heatmap_type
            if heatmap_type == 'amplitude':
                intensity = np.sum(np.abs(summed_signal))
            elif heatmap_type == 'frequency':
                intensity = np.sum(freq_bins * np.abs(summed_signal))
            elif heatmap_type == 'depth':
                intensity = gp[2] * np.sum(np.abs(summed_signal))
            else:
                intensity = np.sum(np.abs(summed_signal))
            
            intensity_map[idx] = intensity
        return intensity_map
