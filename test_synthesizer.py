import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
NUM_SAMPLES = 4000       # 20% of 20,000 steps for test data    
DT = 0.1                 # 100ms per step
OUTPUT_DIR = "data"

# Transition Matrix: [Idle, Accel, Cruise, Decel]
# This guarantees "Normal" behavior flow.
TRANSITION_MATRIX = [
    [0.90, 0.10, 0.00, 0.00], # Idle -> Stay Idle or Start Accelerating
    [0.05, 0.85, 0.09, 0.01], # Accel -> mostly Accel, some Cruise
    [0.01, 0.05, 0.84, 0.10], # Cruise -> mostly Cruise, some Accel or Decel
    [0.20, 0.00, 0.05, 0.75], # Decel -> mostly Decel, eventually Idle
]

class DrivingSimulator:
    def __init__(self):
        self.speed = 0.0
        self.state = 0 # Start at Idle

    def step(self, force_anomaly=False):
        # 1. MARKOV STEP (Decide the "Intent")
        if not force_anomaly:
            self.state = np.random.choice([0, 1, 2, 3], p=TRANSITION_MATRIX[self.state])

        # 2. GAUSSIAN SAMPLING (Generate "Pedal Inputs" based on Intent)
        gas = 0.0
        brake = 0.0
        label = 0  
        
        if self.state == 0:
            gas = np.random.normal(0.0, 0.01)
        elif self.state == 1:
            gas = np.random.normal(0.6, 0.15) 
        elif self.state == 2:
            gas = np.random.normal(0.25, 0.05)
        elif self.state == 3:
            brake = np.random.normal(0.5, 0.1) 

        # --- ANOMALY INJECTION (The "Pedal Misapplication") ---
        # Logic: Physics says "Stop" (State 3), but Driver hits Gas.
        if force_anomaly:
            self.state = 3     # Context: We SHOULD be braking
            brake = 0.0        # ERROR: No brake
            gas = np.random.normal(0.9, 0.05) # ERROR: Full throttle
            label = 1          # 1 = Anomaly

        # Clip pedals to physical limits (0% to 100%)
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # 3. PHYSICS INTEGRATION (The GBSA Logic)
        # Power Factor: How much gas affects acceleration
        # Friction Factor: How much brake affects deceleration
        # Drag: Air resistance (always slows you down slightly)
        
        power = 5.0
        friction = 10.0
        drag = 0.5 + (0.01 * self.speed) # Drag increases with speed

        # Acceleration (a)
        accel = (gas * power) - (brake * friction) - drag

        # Velocity (v = u + at)
        self.speed += accel * DT
        self.speed = max(0.0, self.speed) # Cannot go backwards

        # Return the GBSA vector (Speed, Gas, Brake)
        return [self.speed, gas, brake, label]

# --- MAIN RUN ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Generate TEST Data (With Anomalies)
    print("Generating Test Data (With Anomalies)...")
    sim = DrivingSimulator() # Reset simulator state
    test_data = []
    
    i = 0
    cooldown = 0
    while i < NUM_SAMPLES:
        if cooldown > 0: #gap so no back-to-back anomalies batches.
            test_data.append(sim.step(force_anomaly=False))
            cooldown -= 1
            i += 1
            continue
        #random injection point.
        start_anomaly = (i > 2000 and np.random.rand() < 0.05)

        if start_anomaly:
            anomaly_len = np.random.randint(8, 13)
            cooldown = np.random.randint(80, 100)
            #continuous anomaly steps.
            for _ in range(anomaly_len):
                if i >= NUM_SAMPLES: #edge case
                    break
                # Inject anomaly
                test_data.append(sim.step(force_anomaly=True))
                i += 1
        else:
            #normal step
            test_data.append(sim.step(force_anomaly=False))
            i += 1

    pd.DataFrame(test_data, columns=["speed", "gas", "brake", "label"]).to_csv(f"{OUTPUT_DIR}/test_anomaly.csv", index=False)
    
    print("Done. Data generated in 'data/' folder.")