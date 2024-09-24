# microphone_array.py

import numpy as np

class MicrophoneArray:
    def __init__(self, num_mics, radius):
        self.num_mics = num_mics
        self.radius = radius
        self.positions = self._create_circular_array()
    
    def _create_circular_array(self):
        """
        Creates a circular microphone array on the x-y plane.
        The array faces towards positive z-axis (front side).
        """
        angles = np.linspace(0, 2 * np.pi, self.num_mics, endpoint=False)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        z = np.zeros(self.num_mics)  # All microphones lie on the z=0 plane
        positions = np.vstack((x, y, z)).T
        return positions
    
    def record_signals(self, sources, source_signals, time_array):
        """
        Simulates the signals recorded by each microphone.
        """
        mic_signals = []
        num_samples = len(time_array)
        for mic_pos in self.positions:
            signal = np.zeros(num_samples)
            for source, src_signal in zip(sources, source_signals):
                distance = np.linalg.norm(mic_pos - source.position)
                delay = distance / source.speed_of_sound
                attenuation = 1 / (distance + 1e-6)
                delay_samples = int(delay * source.sampling_rate)
                if delay_samples < num_samples:
                    signal[delay_samples:] += attenuation * src_signal[:num_samples - delay_samples]
            mic_signals.append(signal)
        return mic_signals
