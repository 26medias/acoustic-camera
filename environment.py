import numpy as np

class SimulationEnvironment:
    def __init__(self, grid_size=(200, 200), boundary_type='reflective', wave_speed=1.0, time_step=0.1):
        self.grid_size = grid_size
        self.boundary_type = boundary_type
        self.wave_speed = wave_speed
        self.time_step = time_step

        # Initialize the grids for the simulation
        self.initialize_grid()

    def initialize_grid(self):
        # Current grid representing the wave amplitude at each point
        self.grid = np.zeros(self.grid_size)
        # Grids for the previous and next time steps (needed for the finite difference method)
        self.previous_grid = np.zeros(self.grid_size)
        self.next_grid = np.zeros(self.grid_size)

    def apply_boundaries(self):
        if self.boundary_type == 'reflective':
            # Reflective boundaries: the outer edges mirror the adjacent values
            self.next_grid[0, :] = self.next_grid[1, :]
            self.next_grid[-1, :] = self.next_grid[-2, :]
            self.next_grid[:, 0] = self.next_grid[:, 1]
            self.next_grid[:, -1] = self.next_grid[:, -2]
        elif self.boundary_type == 'open':
            # Open boundaries: waves pass through the edges (could be implemented differently)
            pass
        else:
            raise ValueError(f"Unknown boundary type: {self.boundary_type}")

    def update_environment(self):
        # Wave equation parameters
        c = self.wave_speed
        dt = self.time_step
        dx = 1  # Spatial grid spacing (assuming uniform grid with spacing 1 unit)

        # Calculate the Laplacian (second spatial derivative)
        laplacian = (
            np.roll(self.grid, 1, axis=0) +
            np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) +
            np.roll(self.grid, -1, axis=1) -
            4 * self.grid
        ) / dx**2

        # Update the grid using the finite difference approximation of the wave equation
        self.next_grid = (
            2 * self.grid - self.previous_grid +
            (c * dt)**2 * laplacian
        )

        # Apply boundary conditions
        self.apply_boundaries()

        # Prepare for the next time step
        self.previous_grid, self.grid = self.grid, self.next_grid
        self.next_grid = np.zeros(self.grid_size)  # Reset next_grid to zeros

    def inject_source(self, position, amplitude, frequency, time):
        x, y = position
        # Ensure indices are integers and within bounds
        x = int(np.clip(x, 0, self.grid_size[0] - 1))
        y = int(np.clip(y, 0, self.grid_size[1] - 1))
        # Inject a sinusoidal wave at the specified position
        self.grid[x, y] += amplitude * np.sin(2 * np.pi * frequency * time)

