# main.py

from simulation import Simulation

if __name__ == "__main__":
    # Simulation parameters
    simulation_duration = 1.0  # seconds
    heatmap_type = 'frequency'  # Options: 'depth', 'frequency', 'amplitude'
    
    # Create a Simulation instance
    sim = Simulation(
        duration=simulation_duration,
        heatmap_type=heatmap_type,
        sampling_rate=8000,
        num_mics=5,
        array_radius=1.0,
        lowcut=300,
        highcut=600
    )
    
    # Add sound sources (in front of the array)
    sim.add_sound_source(position=(0.0, 0.0, 2.0), frequency=400, amplitude=1.0)  # 2 meters in front
    sim.add_sound_source(position=(1.0, -0.8, 3.0), frequency=500, amplitude=0.8)  # 3 meters in front
    
    # Run the simulation
    sim.run()
