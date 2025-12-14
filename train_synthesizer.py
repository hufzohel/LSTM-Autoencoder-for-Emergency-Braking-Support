import numpy as np
import pandas as pd
import os

# --- CONFIGURATION (The "Header" definitions) ---
NUM_SAMPLES = 20000      # 20,000 steps = approx 32 minutes of driving data
DT = 0.1                 # Time step (100ms)
OUTPUT_DIR = "data"

STATES = {0: "Idle", 1: "Accel", 2: "Cruise", 3: "Decel"}

# MAROV MATRIX - STATE MACHINE Transition Matrix: P(Next State | Current State)
TRANSITION_MATRIX = [
    [0.90, 0.10, 0.00, 0.00], # From Idle
    [0.05, 0.85, 0.09, 0.01], # From Accel
    [0.01, 0.05, 0.84, 0.10], # From Cruise
    [0.20, 0.00, 0.05, 0.75], # From Decel
]

class DrivingSimulator:
    def __init__(self):
        self.speed = 0.0
        self.state = 0 # Start at Idle

    def step(self):
        """
        Simulates one time step.
        Returns: [speed, gas_position, brake_position]
        """
        self.state = np.random.choice([0, 1, 2, 3], p=TRANSITION_MATRIX[self.state])
        
        gas = 0.0
        brake = 0.0
        
        if self.state == 0:   # Idle
            gas = np.random.normal(0.0, 0.01) # Mean 0, Noise 0.01
        elif self.state == 1: # Accel
            gas = np.random.normal(0.6, 0.15) # Mean 0.6
        elif self.state == 2: # Cruise
            gas = np.random.normal(0.2, 0.05)
        elif self.state == 3: # Decel
            brake = np.random.normal(0.5, 0.1) # Pressing brake
        
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # 3. KINEMATICS:
        # Speed_new = Speed_old + (Gas * Power) - (Brake * Friction) - Drag
        acceleration = (gas * 5.0) - (brake * 10.0) - (0.5 + (0.01 * self.speed))
        self.speed += acceleration * DT
        self.speed = max(0.0, self.speed) 

        return [self.speed, gas, brake]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    sim = DrivingSimulator()
    data_log = []
    for _ in range(NUM_SAMPLES):
        data_log.append(sim.step())

    df_normal = pd.DataFrame(data_log, columns=["speed", "gas", "brake"])
    df_normal.to_csv(f"{OUTPUT_DIR}/train_normal.csv", index=False)

    print("Generate successfully. Data saved to 'data/train_normal.csv'")