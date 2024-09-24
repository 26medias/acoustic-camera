# sound_source.py

import numpy as np

class SoundSource:
    def __init__(self, position, frequency, amplitude, speed_of_sound=343, sampling_rate=8000):
        self.position = np.array(position)  # (x, y, z)
        self.frequency = frequency
        self.amplitude = amplitude
        self.speed_of_sound = speed_of_sound
        self.sampling_rate = sampling_rate
    
    def generate_signal(self, time_array):
        signal = self.amplitude * np.sin(2 * np.pi * self.frequency * time_array)
        return signal
