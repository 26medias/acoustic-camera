import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from environment import SimulationEnvironment
from microphone import Microphone
from fft_analysis import perform_fft
from beamforming import delay_and_sum_beamforming_realtime

class AcousticCameraSimulation:
    def __init__(self,
                 grid_size=(100, 100),
                 boundary_type='reflective',
                 wave_speed=1.0,
                 time_step=0.1,
                 mic_positions=None,
                 sampling_rate=None,
                 animation_speed=20,
                 animation_frames=1000,
                 beamforming_interval=1):
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

        # Beamforming parameters
        self.num_samples = 128  # Number of recent samples for beamforming
        self.scan_grid_size = 50  # Resolution of the scan grid
        self.intensity_map = None  # To store the intensity map
        self.beamforming_interval = beamforming_interval  # Perform beamforming every N frames
        self.frame_counter = 0  # Counter to track frames

        # Initialize the heatmap for beamforming visualization
        self.heatmap = None

    def setup_plot(self):
        # Set up the matplotlib figure with subplots
        self.fig, (self.ax_wave, self.ax_heatmap, self.ax_fft) = plt.subplots(1, 3, figsize=(18, 6))

        # Wave propagation subplot
        self.im = self.ax_wave.imshow(self.env.grid, cmap='viridis', vmin=-1, vmax=1, animated=True)
        self.ax_wave.set_title('2D Wave Propagation Simulation')
        plt.colorbar(self.im, ax=self.ax_wave)

        # Plot microphone positions
        mic_x = [pos[1] for pos in self.mic_positions]  # Note: imshow uses (row, column)
        mic_y = [pos[0] for pos in self.mic_positions]
        self.mic_scatter = self.ax_wave.scatter(mic_x, mic_y, c='red', marker='o', label='Microphones')
        self.ax_wave.legend(loc='upper right')

        # Heatmap subplot for beamforming results
        self.heatmap_im = self.ax_heatmap.imshow(np.zeros((self.scan_grid_size, self.scan_grid_size)),
                                                cmap='hot', extent=(0, self.grid_size[1], self.grid_size[0], 0),
                                                animated=True)
        self.ax_heatmap.set_title('Beamforming Intensity Map')
        self.ax_heatmap.set_xlabel('X Position')
        self.ax_heatmap.set_ylabel('Y Position')
        plt.colorbar(self.heatmap_im, ax=self.ax_heatmap)

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

        # Set up the scan grid for beamforming
        self.setup_scan_grid()


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
        self.frame_counter += 1

        # Inject wave source
        self.inject_wave_source(current_time)

        # Update environment
        self.env.update_environment()
        self.im.set_array(self.env.grid)

        # Record microphone data
        self.record_microphone_data()

        # Update FFT plots
        self.update_fft_plots()

        # Perform beamforming at specified intervals
        if self.frame_counter % self.beamforming_interval == 0:
            self.perform_beamforming_realtime()
            intensity_reshaped = self.intensity_map.reshape(self.X_scan.shape)
            self.heatmap_im.set_data(intensity_reshaped)
            # Optionally adjust color scale
            self.heatmap_im.set_clim(vmin=np.min(intensity_reshaped), vmax=np.max(intensity_reshaped))
        else:
            # Keep the previous intensity map
            pass

        # Return the updated artists
        return [self.im, self.mic_scatter, self.heatmap_im] + self.lines




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
        
        self.perform_beamforming()

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
    
    def perform_beamforming(self):
        from beamforming import delay_and_sum_beamforming

        # Define scan positions along the line (e.g., x positions)
        scan_positions = []
        x_positions = np.linspace(0, self.grid_size[1] - 1, num=100)  # Assuming horizontal line
        y_fixed = self.center_y  # y-coordinate is fixed for 1D imaging

        for x in x_positions:
            scan_positions.append((y_fixed, x))  # Note: positions are (row, column)

        # Perform beamforming
        intensity_map = delay_and_sum_beamforming(
            microphones=self.microphones,
            sound_speed=self.wave_speed,
            scan_positions=scan_positions,
            sampling_rate=self.sampling_rate
        )

        # Plot the intensity map
        plt.figure()
        plt.plot(x_positions, intensity_map)
        plt.title('1D Sound Image using Delay-and-Sum Beamforming')
        plt.xlabel('Position along array (x)')
        plt.ylabel('Intensity')
        plt.show()
    
    def perform_beamforming_realtime(self):

        # Perform beamforming
        self.intensity_map = delay_and_sum_beamforming_realtime(
            microphones=self.microphones,
            sound_speed=self.wave_speed,
            scan_positions=self.scan_positions,
            sampling_rate=self.sampling_rate,
            num_samples=self.num_samples
        )

    
    def setup_scan_grid(self):
        # Define scan positions (x, y) over the simulation area
        x_min, x_max = 0, self.grid_size[1] - 1
        y_min, y_max = 0, self.grid_size[0] - 1
        x_positions = np.linspace(x_min, x_max, self.scan_grid_size)
        y_positions = np.linspace(y_min, y_max, self.scan_grid_size)
        self.X_scan, self.Y_scan = np.meshgrid(x_positions, y_positions)
        self.scan_positions = np.vstack([self.Y_scan.ravel(), self.X_scan.ravel()]).T  # Shape (N, 2)



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
    grid_w = 100
    grid_h = 100

    mic_array = MicArray(grid_width=grid_w, grid_height=grid_h)
    lineMicArray = mic_array.lineArray(count=10, x_start=0, y_start=20, x_end=10, y_end=70)
    arcMicArray = mic_array.arcArray(count=5, radius=50, angle_start=0, angle_end=180, center_x=50, center_y=50)

    # Example usage
    simulation = AcousticCameraSimulation(
        grid_size=(grid_w, grid_h),
        boundary_type='reflective',
        wave_speed=1.0,
        time_step=0.1,
        mic_positions=lineMicArray,
        animation_speed=20,
        animation_frames=1000,
        beamforming_interval=10
    )
    simulation.run_simulation()
