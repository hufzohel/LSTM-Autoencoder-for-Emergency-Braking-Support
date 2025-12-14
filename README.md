# Emergency Braking Support using LSTM Auto-Encoder
### Programming Integration Project CO3101 - HCMUT

**Student:** Trần Nguyễn Mạnh Cường [cite: 7]  
**Instructor:** Ph.D Nguyễn An Khương [cite: 6]  
**University:** Ho Chi Minh City University of Technology (BK TP.HCM) [cite: 2]

---
[Overview and Methodology](https://drive.google.com/file/d/1uxqplWDwh5U7kOGkmcGqhEy6G2JFJq47/view?usp=sharing)
[Visualization and Demo Test](https://drive.google.com/file/d/1OZZ33jBYxILrs9O9GVvVXkhpk8mrsSqZ/view?usp=drive_link)
[HIL Demo](https://drive.google.com/file/d/1OwxwBuc3wrJZh2GQMBWFY5R0bg-5uZ5R/view?usp=drive_link)
---

## 📖 Project Overview

This project implements a **Deep Anomaly Detection** system designed to prevent traffic accidents caused by human error, specifically **pedal misapplication** (mistaking the gas pedal for the brake) 

Unlike traditional Advanced Driver Assistance Systems (ADAS) that rely on rigid, static thresholds, this system uses an **LSTM Auto-Encoder** to learn the temporal patterns of "normal" driving. It calculates a reconstruction error (MSE) in real-time; if the error crosses a dynamic threshold, the behavior is flagged as anomalous, triggering an emergency brake warning.

## 🚀 Key Features

* **Unsupervised Learning:** Does not require labeled accident data, which is scarce and dangerous to collect. It trains exclusively on normal driving data
* **Temporal Awareness:** Uses a sliding window of **3 seconds (30 time steps)** to understand the context of vehicle dynamics rather than just instantaneous values
* **Physics-Aware Data Synthesis:** Generates high-fidelity synthetic training data using Markov Chains combined with Kinematic equations ($F=ma$, Friction, Drag)
* **Hardware-in-the-Loop (HIL):** A real-time demo using an **Arduino Mega 2560** to simulate pedal inputs and a Python host for inference

## 🛠️ System Architecture

### 1. Data Pipeline & Synthesis
Since real-world pedal misapplication data is rare, we synthesized data using a hybrid statistical-physical model:
* **Markov Chain:** Defines 4 driving states (**Idle, Acceleration, Cruise, Deceleration**) and transition probabilities to ensure logical flow
* **Kinematics:** Enforces physical constraints (Inertia, Drag, Friction) so the vehicle speed follows realistic curves ($v = u + at$)
* **Anomalies:** Test data includes injected "Pedal Misapplication" events where the gas is pressed while the physics dictate a stop state

### 2. The Model (LSTM Auto-Encoder)
* **Input:** Vector of shape `(Batch, 30, 3)` representing `[Speed, Gas Position, Brake Position]`[cite: 416, 393].
* **Encoder:** Compresses the 3-second window into a **Latent Vector (dim=4)**[cite: 453]. This bottleneck forces the model to learn high-level driving concepts.
* **Decoder:** Reconstructs the original sequence
* **Loss Function:** Mean Squared Error (MSE)

### 3. Hardware-in-the-Loop (HIL)
* **Arduino Mega 2560:** Reads analog values from 3 Potentiometers (Simulating Gas, Brake, Speed)
* **Python Host:** Reads serial data, runs the PyTorch model, calculates MSE, and sends a "DANGER" signal back to Arduino if the threshold is breached
* **Feedback:** An LED on the Arduino flashes when an anomaly is detected

## 💻 Tech Stack

* **Language:** Python 
* **Deep Learning:** PyTorch (Inferred from code snippets: `torch.FloatTensor`, `torch.optim`)[cite: 436, 478]. *Note: Report mentions Tensorflow/Keras in section 2.5, but implementation details use PyTorch.*
* **Data Processing:** NumPy, Pandas
* **Hardware Interface:** PySerial
* **Hardware:** Arduino C++

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```

2.  **Install Dependencies:**
    ```bash
    pip install numpy pandas torch pyserial
    ```

3.  **Hardware Setup:**
    * Connect 3 Potentiometers to Analog Pins `A0` (Gas), `A1` (Brake), and `A2` (Speed) on an Arduino Mega
    * Upload the provided Arduino sketch to the board.

## 🏃 Usage

### 1. Data Synthesis
Generate the training (normal) and testing (anomaly) datasets.
```bash
python data_synthesis.py
