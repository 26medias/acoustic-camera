# simulation.py

import numpy as np
from microphone_array import MicrophoneArray
from sound_source import SoundSource
from signal_processing import SignalProcessor
from visualization import Visualizer

class Simulation:
    def __init__(self, duration, heatmap_type, sampling_rate=8000, num_mics=50, array_radius=1.0, lowcut=300, highcut=600):
        self.duration = duration
        self.heatmap_type = heatmap_type  # 'depth', 'frequency', or 'amplitude'
        self.sampling_rate = sampling_rate
        self.num_mics = num_mics
        self.array_radius = array_radius
        self.lowcut = lowcut
        self.highcut = highcut
        self.time_array = np.linspace(0, duration, int(sampling_rate * duration))
        self.sources = []
        self.mic_array = MicrophoneArray(num_mics, array_radius)
        self.signal_processor = SignalProcessor(sampling_rate, lowcut, highcut)
        self.visualizer = Visualizer()
        self.mic_signals = []
    
    def add_sound_source(self, position, frequency, amplitude):
        source = SoundSource(position, frequency, amplitude)
        self.sources.append(source)
    
    def run(self):
        # Generate source signals
        source_signals = [source.generate_signal(self.time_array) for source in self.sources]
        
        # Simulate microphone recordings
        self.mic_signals = self.mic_array.record_signals(self.sources, source_signals, self.time_array)
        
        # Process signals
        filtered_signals = self.signal_processor.filter_signals(self.mic_signals)
        
        # Perform localization
        intensity_map = self.signal_processor.perform_beamforming(
            self.mic_array.positions,
            filtered_signals,
            self.heatmap_type
        )
        
        # Visualize results
        self.visualizer.plot_heatmap(
            intensity_map,
            self.mic_array.positions,
            self.sources,
            self.heatmap_type
        )
