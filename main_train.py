import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle # To save the Scaler and Threshold
from sklearn.preprocessing import MinMaxScaler
from model import LSTM_Autoencoder

# --- CONFIGURATION ---
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "lstm_model.pth"
META_PATH = "train_meta.pkl" # Saves Scaler and Threshold

def create_sliding_windows(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

def train():
    # 1. LOAD NORMAL DATA (Train on Normal Only)
    print("Loading Training Data...")
    try:
        df_train = pd.read_csv("data/train_normal.csv")
    except FileNotFoundError:
        print("Error: data/train_normal.csv not found.")
        return

    train_values = df_train[["speed", "gas", "brake"]].values

    # 2. FIT SCALER
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)

    # 3. PREPARE TENSORS
    X_train = create_sliding_windows(train_scaled, WINDOW_SIZE)
    X_train_tensor = torch.FloatTensor(X_train)

    # 4. INITIALIZE MODEL
    model = LSTM_Autoencoder(input_dim=3, hidden_dim=64, latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5. TRAINING LOOP
    model.train()
    num_batches = len(X_train_tensor) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = 0
        indices = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[indices]

        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            batch = X_train_shuffled[start:end]

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/num_batches:.5f}")

    # 6. CALCULATE THRESHOLD (The "Normalcy" Limit)
    print("Calculating")
    model.eval()
    with torch.no_grad():
        rec_train = model(X_train_tensor)
        # Error per sequence
        train_mse = np.mean(np.power(X_train_tensor.numpy() - rec_train.numpy(), 2), axis=(1, 2))
    
    # Threshold = Mean + 1 Std Devs
    # used to be 4.0, 3.0
    threshold = np.mean(train_mse) + 1.0 * np.std(train_mse)
    print(f"Threshold set to: {threshold:.5f}")
    
    # Save Weights
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save Scaler and Threshold (The "Bridge")
    with open(META_PATH, 'wb') as f:
        pickle.dump({'scaler': scaler, 'threshold': threshold}, f)
        
    print("Training Complete. Files saved.")

if __name__ == "__main__":
    train()