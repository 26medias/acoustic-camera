class Microphone:
    def __init__(self, position, sampling_rate=1.0):
        self.position = position  # Tuple of (x, y) coordinates
        self.sampling_rate = sampling_rate  # Samples per unit time
        self.recorded_amplitudes = []  # List to store recorded amplitudes

    def record(self, amplitude):
        # Append the amplitude to the recorded data
        self.recorded_amplitudes.append(amplitude)

    def reset_data(self):
        self.recorded_amplitudes = []
