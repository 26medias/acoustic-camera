import numpy as np

def delay_and_sum_beamforming(microphones, sound_speed, scan_positions, sampling_rate):
    """
    Perform delay-and-sum beamforming.

    Parameters:
    - microphones: list of Microphone objects with recorded amplitudes and positions.
    - sound_speed: Speed of sound in the medium.
    - scan_positions: List of positions (x, y) along the array to scan.
    - sampling_rate: Sampling rate of the recordings.

    Returns:
    - intensity_map: List of summed amplitudes corresponding to each scan position.
    """
    num_samples = len(microphones[0].recorded_amplitudes)
    intensity_map = []

    for scan_pos in scan_positions:
        # Calculate delays for each microphone
        delays = []
        for mic in microphones:
            distance = np.linalg.norm(np.array(scan_pos) - np.array(mic.position))
            delay = distance / sound_speed
            delays.append(delay)

        # Convert delays to sample indices
        delay_samples = [int(delay * sampling_rate) for delay in delays]

        # Align and sum signals
        summed_signal = np.zeros(num_samples)
        for mic, d in zip(microphones, delay_samples):
            shifted_signal = np.roll(mic.recorded_amplitudes, -d)
            summed_signal += shifted_signal[:num_samples]

        # Calculate intensity (e.g., RMS of the summed signal)
        intensity = np.sqrt(np.mean(summed_signal ** 2))
        intensity_map.append(intensity)

    return intensity_map


def delay_and_sum_beamforming_realtime(microphones, sound_speed, scan_positions, sampling_rate, num_samples):
    """
    Perform delay-and-sum beamforming for real-time visualization.

    Parameters:
    - microphones: list of Microphone objects with recorded amplitudes and positions.
    - sound_speed: Speed of sound in the medium.
    - scan_positions: Array of shape (N, 2) for N scan positions.
    - sampling_rate: Sampling rate of the recordings.
    - num_samples: Number of recent samples to use.

    Returns:
    - intensity_map: Array of intensities corresponding to each scan position.
    """

    num_mics = len(microphones)
    num_scan_positions = scan_positions.shape[0]

    # Collect recent samples from each microphone
    mic_signals = np.zeros((num_mics, num_samples))
    for idx, mic in enumerate(microphones):
        if len(mic.recorded_amplitudes) >= num_samples:
            mic_signals[idx] = mic.recorded_amplitudes[-num_samples:]
        else:
            # Pad with zeros if not enough samples
            mic_signals[idx, -len(mic.recorded_amplitudes):] = mic.recorded_amplitudes

    # Precompute microphone positions
    mic_positions = np.array([mic.position for mic in microphones])

    # Compute delays for each scan position and microphone
    # Distance matrix of shape (num_scan_positions, num_mics)
    distances = np.linalg.norm(scan_positions[:, np.newaxis, :] - mic_positions[np.newaxis, :, :], axis=2)
    delays = distances / sound_speed  # Time delays in seconds
    delay_samples = (delays * sampling_rate).astype(int)  # Convert to sample indices

    # Ensure delay_samples are within bounds
    max_delay = delay_samples.max()
    if max_delay >= num_samples:
        # Adjust num_samples or clip delays
        delay_samples = np.clip(delay_samples, 0, num_samples - 1)

    # Initialize intensity map
    intensity_map = np.zeros(num_scan_positions)

    # For each scan position, compute the summed signal
    for idx in range(num_scan_positions):
        shifted_signals = np.zeros((num_mics, num_samples))
        for m in range(num_mics):
            d = delay_samples[idx, m]
            shifted_signals[m] = np.roll(mic_signals[m], -d)
        summed_signal = shifted_signals.sum(axis=0)
        # Calculate intensity (e.g., RMS of the summed signal)
        intensity = np.sqrt(np.mean(summed_signal ** 2))
        intensity_map[idx] = intensity

    return intensity_map
