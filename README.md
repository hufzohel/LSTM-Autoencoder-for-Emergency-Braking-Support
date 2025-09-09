# LSTM-Autoencoder-for-Emergency-Braking-Support
The project aim to solve or reduce the number of traffic accidents caused by human-error especially mistaking the gas-pedal for the brake-pedal by implementing LSTM autoencoder using Tensorflow/Keras to read vehicle's sensor data: brake-pedal position, gas-pedal position and vehicle speed. The goal is to learn the pattern in such errors and support making the right decision: to brake or not to brake.
## 📌Table of Contents
*  [Overview](🔍Overview)
*  [Tech Stack](⚙️Tech-Stack)
*  [Methodology](🧠Methodology)
   *  [Dataset](📁Dataset)
   *  [Architecture](🏗️Architecture)
*  [Result](📊Result)
*  [Setup](🚀Setup)
*  [Authors](👤Authors)
---
## 🔍Overview
* Model includes: LSTM, Autoencoder
* Sequential data analysis on vehicle's sensors
* Emergency Braking Support based on temporal pattern
* Data is synthesized
## ⚙️Tech-Stack
* Language: Python
* Libraries: Tensorflow/Keras, Numpy, Pandas.
* Tools: Google Colab, Github Projects
## 🧠Methodology
### 📁Dataset
Due to the lack of authentic data, the data are synthesized:
* Uses Python (Numpy, Pandas) to synthesizes sequential Data in a period of 3 seconds - that mimic "Normal" and "Abnormal" driving behavior
   * Normal: start and stop gently, stable speed level.
   * Abnormal: abrupt gas-pressed, unstable/spike in speed level.
* Data features: Gas-pedal position, Brake-pedal position, Vehicle speed.
### 🏗️Architecture
LSTM Autoencoder for Emergency Braking Support
* Autoencoder: a neural network designed to learn compressed representations of data.
* LSTM: a type of RNN aimed at mitigating vanishing gradient.
* Braking Decision based on sequential data from vehicle sensors:
   * Data from vehicle sensors are fed every 1/5 of a second. 
   * Temporal Pattern: Abrupt spike in gas pedal position coupled with sudden burst in speed.
## 📊Result

## 🚀Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/hufzohel/LSTM-Autoencoder-for-Emergency-Braking-Support.git
   ```
## 👤Authors
Trần Nguyễn Mạnh Cường - MSSV:2210446

