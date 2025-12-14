# Emergency Braking Support using LSTM Auto-Encoder
### Programming Integration Project CO3101 - HCMUT

[cite_start]**Student:** Trần Nguyễn Mạnh Cường [cite: 7]  
[cite_start]**Instructor:** Ph.D Nguyễn An Khương [cite: 6]  
[cite_start]**University:** Ho Chi Minh City University of Technology (BK TP.HCM) [cite: 2]

---

## 📖 Project Overview

[cite_start]This project implements a **Deep Anomaly Detection** system designed to prevent traffic accidents caused by human error, specifically **pedal misapplication** (mistaking the gas pedal for the brake)[cite: 18, 51]. 

[cite_start]Unlike traditional Advanced Driver Assistance Systems (ADAS) that rely on rigid, static thresholds[cite: 20], this system uses an **LSTM Auto-Encoder** to learn the temporal patterns of "normal" driving. [cite_start]It calculates a reconstruction error (MSE) in real-time; if the error crosses a dynamic threshold, the behavior is flagged as anomalous, triggering an emergency brake warning[cite: 22, 24].

## 🚀 Key Features

* **Unsupervised Learning:** Does not require labeled accident data, which is scarce and dangerous to collect. [cite_start]It trains exclusively on normal driving data[cite: 378, 400].
* [cite_start]**Temporal Awareness:** Uses a sliding window of **3 seconds (30 time steps)** to understand the context of vehicle dynamics rather than just instantaneous values[cite: 413, 415].
* [cite_start]**Physics-Aware Data Synthesis:** Generates high-fidelity synthetic training data using Markov Chains combined with Kinematic equations ($F=ma$, Friction, Drag)[cite: 146, 147].
* [cite_start]**Hardware-in-the-Loop (HIL):** A real-time demo using an **Arduino Mega 2560** to simulate pedal inputs and a Python host for inference[cite: 764].

## 🛠️ System Architecture

### 1. Data Pipeline & Synthesis
Since real-world pedal misapplication data is rare, we synthesized data using a hybrid statistical-physical model:
* [cite_start]**Markov Chain:** Defines 4 driving states (**Idle, Acceleration, Cruise, Deceleration**) and transition probabilities to ensure logical flow[cite: 92, 125].
* [cite_start]**Kinematics:** Enforces physical constraints (Inertia, Drag, Friction) so the vehicle speed follows realistic curves ($v = u + at$)[cite: 141, 144].
* [cite_start]**Anomalies:** Test data includes injected "Pedal Misapplication" events where the gas is pressed while the physics dictate a stop state[cite: 301].

### 2. The Model (LSTM Auto-Encoder)
* [cite_start]**Input:** Vector of shape `(Batch, 30, 3)` representing `[Speed, Gas Position, Brake Position]`[cite: 416, 393].
* [cite_start]**Encoder:** Compresses the 3-second window into a **Latent Vector (dim=4)**[cite: 453]. This bottleneck forces the model to learn high-level driving concepts.
* [cite_start]**Decoder:** Reconstructs the original sequence[cite: 462].
* [cite_start]**Loss Function:** Mean Squared Error (MSE)[cite: 465].

### 3. Hardware-in-the-Loop (HIL)
* [cite_start]**Arduino Mega 2560:** Reads analog values from 3 Potentiometers (Simulating Gas, Brake, Speed)[cite: 764].
* [cite_start]**Python Host:** Reads serial data, runs the PyTorch model, calculates MSE, and sends a "DANGER" signal back to Arduino if the threshold is breached[cite: 1029].
* [cite_start]**Feedback:** An LED on the Arduino flashes when an anomaly is detected[cite: 767].

## 💻 Tech Stack

* [cite_start]**Language:** Python [cite: 56]
* [cite_start]**Deep Learning:** PyTorch (Inferred from code snippets: `torch.FloatTensor`, `torch.optim`)[cite: 436, 478]. *Note: Report mentions Tensorflow/Keras in section 2.5, but implementation details use PyTorch.*
* [cite_start]**Data Processing:** NumPy, Pandas[cite: 57].
* [cite_start]**Hardware Interface:** PySerial[cite: 766].
* [cite_start]**Hardware:** Arduino C++[cite: 54].

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
    * [cite_start]Connect 3 Potentiometers to Analog Pins `A0` (Gas), `A1` (Brake), and `A2` (Speed) on an Arduino Mega[cite: 789, 791, 793].
    * Upload the provided Arduino sketch to the board.

## 🏃 Usage

### 1. Data Synthesis
Generate the training (normal) and testing (anomaly) datasets.
```bash
python data_synthesis.py
