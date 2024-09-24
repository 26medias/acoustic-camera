# visualization.py

import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def plot_heatmap(self, intensity_map, mic_positions, sources, heatmap_type):
        # Assume intensity_map is a 3D array: (x, y, z)
        x_size, y_size, z_size = intensity_map.shape
        z_slices = [int(z_size / 2)]  # Middle depth slice
        
        for z_idx in z_slices:
            intensity_slice = intensity_map[:, :, z_idx]
            plt.figure(figsize=(8, 6))
            plt.imshow(
                np.flipud(intensity_slice.T),
                extent=[-3, 3, -3, 3],
                cmap='inferno',
                aspect='auto'
            )
            plt.title(f'{heatmap_type.capitalize()} Heatmap at Depth Index {z_idx}')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.colorbar(label='Intensity')
            # Plot microphone positions
            plt.scatter(
                mic_positions[:, 0],
                mic_positions[:, 1],
                c='cyan',
                marker='^',
                label='Microphones'
            )
            # Plot source positions
            for source in sources:
                plt.scatter(
                    source.position[0],
                    source.position[1],
                    c='green',
                    marker='*',
                    s=100,
                    label='Source'
                )
            plt.legend()
            plt.show()
