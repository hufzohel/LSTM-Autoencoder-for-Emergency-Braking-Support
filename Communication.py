import serial
import time
import numpy as np
from collections import deque
import torch
import pickle

# --- IMPORT MODEL DEFINITION ---
from model import LSTM_Autoencoder

# ===============================
# CONFIGURATION
# ===============================
PORT = 'COM3'
BAUD_RATE = 9600

WINDOW_SIZE = 30          # 3 seconds @ 100ms
MODEL_PATH = 'lstm_model.pth'
META_PATH = 'train_meta.pkl'

# ===============================
# LOAD MODEL + METADATA
# ===============================
print("Loading model and metadata...")

with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
    scaler = meta['scaler']
    threshold = meta['threshold']

model = LSTM_Autoencoder(input_dim=3, hidden_dim=64, latent_dim=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

print(f"✅ Model loaded | Threshold = {threshold:.5f}")

# ===============================
# SIMPLE SMOOTHING FILTER
# ===============================
class SmoothSensor:
    def __init__(self, window_size=5):
        self.data = deque(maxlen=window_size)

    def update(self, value):
        self.data.append(value)
        return sum(self.data) / len(self.data)

gas_filter = SmoothSensor()
brake_filter = SmoothSensor()
speed_filter = SmoothSensor()

# ===============================
# SLIDING WINDOW BUFFERS
# ===============================
window_buffer = deque(maxlen=WINDOW_SIZE)     # stores [speed, gas, brake]
speed_buffer = deque(maxlen=WINDOW_SIZE)      # used for motion gate

# ===============================
# CONNECT TO ARDUINO
# ===============================
try:
    arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"✅ Connected to Arduino on {PORT}")
except:
    print("❌ Failed to connect to Arduino")
    exit()

# ===============================
# MAIN LOOP
# ===============================
print("Starting live inference...")
while True:
    try:
        if arduino.in_waiting == 0:
            continue

        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')

        if len(parts) < 3:
            continue

        # --------------------------------
        # 1. READ RAW SENSOR DATA
        # --------------------------------
        raw_gas = int(parts[0])
        raw_brake = int(parts[1])
        raw_speed = int(parts[2])

        # --------------------------------
        # 2. SMOOTH SENSOR DATA
        # --------------------------------
        gas = gas_filter.update(raw_gas)
        brake = brake_filter.update(raw_brake)
        speed = speed_filter.update(raw_speed)

        # --------------------------------
        # 3. APPEND TO WINDOW BUFFER
        # --------------------------------
        window_buffer.append([speed, gas, brake])
        speed_buffer.append(speed)

        # Wait until buffer is full
        if len(window_buffer) < WINDOW_SIZE:
            print("⏳ Buffering...")
            continue

        # --------------------------------
        # 4. SCALE USING TRAINING SCALER
        # --------------------------------
        window_np = np.array(window_buffer)
        window_scaled = scaler.transform(window_np)

        input_tensor = torch.FloatTensor(window_scaled).unsqueeze(0)
        # shape: [1, 30, 3]

        # --------------------------------
        # 5. AUTOENCODER INFERENCE
        # --------------------------------
        with torch.no_grad():
            reconstructed = model(input_tensor)
            mse = np.mean(
                (input_tensor.numpy() - reconstructed.numpy()) ** 2
            )

        # --------------------------------
        # 6. DECISION LOGIC (GATED)
        # --------------------------------
        mean_speed = np.mean(speed_buffer)
        is_anomaly = (mse > threshold) and (mean_speed > 0.05)

        # --------------------------------
        # 7. SEND FEEDBACK TO ARDUINO
        # --------------------------------
        if is_anomaly:
            arduino.write(b'1')
            status = "⚠️ DANGER"
        else:
            arduino.write(b'0')
            status = "Safe"

        print(
            f"Speed:{speed:6.1f} | Gas:{gas:6.1f} | Brake:{brake:6.1f} "
            f"| MSE:{mse:.5f} | {status}"
        )

    except KeyboardInterrupt:
        print("\nStopping...")
        arduino.close()
        break

    except Exception as e:
        print("Error:", e)
