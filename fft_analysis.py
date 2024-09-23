import numpy as np

def perform_fft(amplitude_data, sampling_rate):
    # Number of samples
    N = len(amplitude_data)
    # Perform FFT
    fft_values = np.fft.fft(amplitude_data)
    # Compute the frequencies corresponding to the FFT values
    freq = np.fft.fftfreq(N, d=1.0 / sampling_rate)
    # Only take the positive frequencies
    idx = np.arange(N // 2)
    freq = freq[idx]
    fft_values = fft_values[idx]
    # Return frequencies and their corresponding amplitudes
    return freq, np.abs(fft_values)
