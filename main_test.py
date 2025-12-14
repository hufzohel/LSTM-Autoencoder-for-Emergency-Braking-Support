import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- IMPORTS ---
from model import LSTM_Autoencoder

# --- CONFIGURATION ---
WINDOW_SIZE = 30
MODEL_PATH = "lstm_model.pth"
META_PATH = "train_meta.pkl"

def create_sliding_windows(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

def test():
    # 1. LOAD THE "BRIDGE" (Scaler & Threshold from Training)
    print("Loading Model and Metadata...")
    try:
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
            scaler = meta['scaler']       # The Unit Converter
            threshold = meta['threshold'] # The "Danger Line"
            
        # Load the Brain
        model = LSTM_Autoencoder(input_dim=3, hidden_dim=64, latent_dim=4)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Switch to Read-Only mode
        
    except FileNotFoundError:
        print("Error: Model files not found. Did you run 'train.py'?")
        return

    print(f"Loaded Threshold: {threshold:.5f}")

    # 2. LOAD TEST DATA
    try:
        df_test = pd.read_csv("data/test_anomaly.csv")
    except FileNotFoundError:
        print("Error: data/test_anomaly.csv not found.")
        return

    # Extract Inputs
    test_values = df_test[["speed", "gas", "brake"]].values
    
    # Extract Labels (Answer Key)
    if "label" in df_test.columns:
        test_labels = df_test["label"].values 
        has_labels = True
    else:
        print("Warning: No 'label' column found. Cannot calculate F1 Score.")
        test_labels = np.zeros(len(df_test)) # Dummy labels to prevent crash
        has_labels = False

    # 3. NORMALIZE (Using the TRAINING Scaler)
    # We must judge the test data by the same standards as training data
    test_scaled = scaler.transform(test_values)

    # Keep track of speed
    speed_values = test_values[:, 0]
    speed_windows = create_sliding_windows(speed_values, WINDOW_SIZE)
    mean_speed = speed_windows.mean(axis=1)   # shape: (num_windows,)

    # 4. CREATE WINDOWS
    X_test = create_sliding_windows(test_scaled, WINDOW_SIZE)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Align labels to the end of the window
    # Create sliding windows of labels
    label_windows = create_sliding_windows(test_labels, WINDOW_SIZE)
    y_test = (label_windows.sum(axis=1) >= 8).astype(int)

    assert len(X_test) == len(y_test) == len(mean_speed)


    # 5. RUN THE "CAR COMPUTER" (Inference)
    print(f"Running Inference on {len(X_test)} windows...")
    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        # Difference between Reality and Model Expectation
        test_mse = np.mean(np.power(X_test_tensor.numpy() - reconstructed.numpy(), 2), axis=(1, 2))

    predictions = ((test_mse > threshold) & (mean_speed > 0.05)).astype(int)

    # 7. CALCULATE SCORE (If labels exist)
    if has_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
        
        print("\n--- REPORT CARD ---")
        print(f"Precision: {precision:.2f} (Did we falsely scream?)")
        print(f"Recall:    {recall:.2f} (Did we catch the crash?)")
        print(f"F1 Score:  {f1:.2f} (Overall Grade)")
        
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(f"True Normal: {cm[0][0]} | False Alarm: {cm[0][1]}")
        print(f"Missed Crash: {cm[1][0]} | Caught Crash: {cm[1][1]}")
    else:
        print("\nSkipping F1 Score (No Labels found).")

if __name__ == "__main__":
    test()