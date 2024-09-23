import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from environment import SimulationEnvironment
from microphone import Microphone
from fft_analysis import perform_fft

class AcousticCameraSimulation:
    def __init__(self,
                 grid_size=(100, 100),
                 boundary_type='reflective',
                 wave_speed=1.0,
                 time_step=0.1,
                 mic_positions=None,
                 sampling_rate=None,
                 animation_speed=20,
                 animation_frames=1000):
        # Simulation environment parameters
        self.grid_size = grid_size
        self.boundary_type = boundary_type
        self.wave_speed = wave_speed
        self.time_step = time_step

        # Initialize the simulation environment
        self.env = SimulationEnvironment(
            grid_size=self.grid_size,
            boundary_type=self.boundary_type,
            wave_speed=self.wave_speed,
            time_step=self.time_step
        )

        # Wave source position (center of the grid)
        self.center_x = self.grid_size[0] // 2
        self.center_y = self.grid_size[1] // 2

        # Microphone positions
        if mic_positions is None:
            # Default to three microphones in a horizontal line
            self.mic_positions = [(self.center_x, self.center_y - 30),
                                  (self.center_x, self.center_y),
                                  (self.center_x, self.center_y + 30)]
        else:
            self.mic_positions = mic_positions

        # Initialize microphones
        self.microphones = [Microphone(position=pos) for pos in self.mic_positions]

        # Sampling rate
        self.sampling_rate = sampling_rate or (1.0 / self.time_step)

        # Animation parameters
        self.animation_speed = animation_speed
        self.animation_frames = animation_frames

        # Initialize other variables
        self.fig = None
        self.ax_wave = None
        self.ax_fft = None
        self.im = None
        self.mic_scatter = None
        self.lines = []
        self.slider_freq = None
        self.slider_amp = None
        self.frequency = 2.0  # Default frequency
        self.amplitude = 1.0  # Default amplitude

    def setup_plot(self):
        # Set up the matplotlib figure with subplots
        self.fig, (self.ax_wave, self.ax_fft) = plt.subplots(1, 2, figsize=(12, 6))

        # Wave propagation subplot
        self.im = self.ax_wave.imshow(self.env.grid, cmap='viridis', vmin=-1, vmax=1, animated=True)
        self.ax_wave.set_title('2D Wave Propagation Simulation')
        plt.colorbar(self.im, ax=self.ax_wave)

        # Plot microphone positions
        mic_x = [pos[1] for pos in self.mic_positions]  # Note: imshow uses (row, column)
        mic_y = [pos[0] for pos in self.mic_positions]
        self.mic_scatter = self.ax_wave.scatter(mic_x, mic_y, c='red', marker='o', label='Microphones')
        self.ax_wave.legend(loc='upper right')

        # FFT subplot
        colors = ['blue', 'green', 'orange', 'purple', 'cyan']  # Extend if more microphones are added
        self.lines = []
        for idx, mic in enumerate(self.microphones):
            color = colors[idx % len(colors)]
            line, = self.ax_fft.plot([], [], color=color, label=f'Microphone {idx+1}')
            self.lines.append(line)

        self.ax_fft.set_title('Real-Time FFT of Microphone Recordings')
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Amplitude')
        self.ax_fft.set_xlim(0, self.sampling_rate / 2)
        self.ax_fft.set_ylim(0, 50)  # Adjust based on expected amplitude
        self.ax_fft.legend()

        # Adjust the layout to make room for sliders
        plt.subplots_adjust(bottom=0.25)

        # Create axes for sliders
        ax_freq = plt.axes([0.15, 0.1, 0.65, 0.03])
        ax_amp = plt.axes([0.15, 0.05, 0.65, 0.03])

        # Create sliders
        self.slider_freq = Slider(ax_freq, 'Frequency', 0.1, 5.0, valinit=self.frequency)
        self.slider_amp = Slider(ax_amp, 'Amplitude', 0.1, 2.0, valinit=self.amplitude)

        # Connect sliders to update functions
        self.slider_freq.on_changed(self.update_frequency)
        self.slider_amp.on_changed(self.update_amplitude)

    def update_frequency(self, val):
        self.frequency = self.slider_freq.val

    def update_amplitude(self, val):
        self.amplitude = self.slider_amp.val

    def inject_wave_source(self, current_time):
        # Inject continuous wave at the center
        self.env.inject_source(
            position=(self.center_x, self.center_y),
            amplitude=self.amplitude,
            frequency=self.frequency,
            time=current_time
        )

    def record_microphone_data(self):
        # Record amplitudes at microphone positions
        for mic in self.microphones:
            x, y = mic.position
            x = int(np.clip(x, 0, self.env.grid_size[0] - 1))
            y = int(np.clip(y, 0, self.env.grid_size[1] - 1))
            amplitude_at_mic = self.env.grid[x, y]
            mic.record(amplitude_at_mic)

    def update_fft_plots(self):
        # Perform FFT for each microphone and update the plot
        window_size = 128
        for idx, mic in enumerate(self.microphones):
            if len(mic.recorded_amplitudes) >= window_size:
                # Get the latest window_size samples
                recent_data = mic.recorded_amplitudes[-window_size:]
                freq, fft_values = perform_fft(recent_data, self.sampling_rate)
                # Update the corresponding line in the FFT plot
                self.lines[idx].set_data(freq, fft_values)
            else:
                # Not enough data yet, set empty data
                self.lines[idx].set_data([], [])

        # Adjust FFT plot limits if necessary
        self.ax_fft.relim()
        self.ax_fft.autoscale_view()

    def update(self, frame):
        current_time = frame * self.env.time_step

        # Inject wave source
        self.inject_wave_source(current_time)

        # Update environment
        self.env.update_environment()
        self.im.set_array(self.env.grid)

        # Record microphone data
        self.record_microphone_data()

        # Update FFT plots
        self.update_fft_plots()

        # Return the updated artists
        return [self.im, self.mic_scatter] + self.lines

    def run_simulation(self):
        # Set up the plot
        self.setup_plot()

        # Create the animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.animation_frames,
            interval=self.animation_speed,
            blit=True,
            repeat=False
        )

        # Display the animation
        plt.show()

        # After the simulation, perform analysis
        self.post_simulation_analysis()

    def post_simulation_analysis(self):
        # Print recorded amplitudes (optional)
        for i, mic in enumerate(self.microphones, start=1):
            print(f"Microphone {i} at position {mic.position} recorded {len(mic.recorded_amplitudes)} samples.")
        return
        # Perform FFT analysis for each microphone
        for i, mic in enumerate(self.microphones, start=1):
            freq, fft_values = perform_fft(mic.recorded_amplitudes, self.sampling_rate)
            # Plot the FFT results
            plt.figure()
            plt.plot(freq, fft_values)
            plt.title(f'FFT of Microphone {i} Recordings')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.show()


import math

class MicArray:
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height

    def lineArray(self, count: int, x_start: int, y_start: int, x_end: int, y_end: int):
        coords = []
        if count < 2:
            return [(x_start, y_start)]  # Single point if count is 1
        
        x_spacing = (x_end - x_start) / (count - 1)
        y_spacing = (y_end - y_start) / (count - 1)

        for i in range(count):
            x = x_start + i * x_spacing
            y = y_start + i * y_spacing
            coords.append((int(x), int(y)))
        
        return coords

    def arcArray(self, count: int, radius: int, angle_start: float, angle_end: float, center_x: int = 0, center_y: int = 0):
        coords = []
        angle_step = (angle_end - angle_start) / (count - 1) if count > 1 else 0
        for i in range(count):
            angle = math.radians(angle_start + i * angle_step)
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            coords.append((x, y))
        return coords


if __name__ == '__main__':
    grid_w = 200
    grid_h = 200

    mic_array = MicArray(grid_width=grid_w, grid_height=grid_h)
    lineMicArray = mic_array.lineArray(count=20, x_start=10, y_start=20, x_end=25, y_end=80)
    arcMicArray = mic_array.arcArray(count=5, radius=50, angle_start=0, angle_end=180, center_x=50, center_y=50)

    # Example usage
    simulation = AcousticCameraSimulation(
        grid_size=(grid_w, grid_h),
        boundary_type='reflective',
        wave_speed=1.0,
        time_step=0.1,
        mic_positions=lineMicArray,
        animation_speed=20,
        animation_frames=5000
    )
    simulation.run_simulation()
