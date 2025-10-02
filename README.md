# EEG-EMG-Hands 🤖🧠  
Control a robotic hand using muscle signals (sEMG) and brainwaves (EEG) with OpenBCI and Brainflow.

This project uses an OpenBCI Cyton/Daisy board, but is compatible with any [supported Brainflow board](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html).

https://github.com/user-attachments/assets/e529cad6-1202-485f-9131-40d391461205

Unlike gesture recognition, this project allows you to control each finger individually.

---

## 🚀 How to Use

### 1. Set Up the Python Environment

Recommended: Python 3.10  
Create and activate a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

More info: [Python venv documentation](https://docs.python.org/3/library/venv.html)

---

### 2. Stream EXG (EEG/EMG) Data

Open `exg/brainflow_server.py` and change `BOARD_ID` to your board (e.g. 
BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD)

**On macOS:**
```bash
sh stream_brainflow.sh
```

If you're on another OS or the script fails, run the Python commands inside `stream_brainflow.sh` manually.

---

### 3. Stream Prompt Data

**On macOS:**
```bash
sh stream_prompt.sh
```

This streams prompt (command) labels used for supervised learning.

If you're on another OS or the script fails, run the Python commands inside `stream_prompt.sh` manually.

---

### 4. Connect the ESP32 Hand (Work in Progress)

1. Flash `esp32/hand_esp.ino` to your ESP32.
2. Connect the servo controller to the prosthetic hand and ESP32.
3. Run:
```bash
sh hand_prompt.sh
```

---

### 5. Record Data to CSV

In `record_lsl_csv.py`, set:

```python
SESSION_DIR = "data/s_MM_DD_YY"
```

Then run:
```bash
python record_lsl_csv.py
```

To record multiple sessions in one day (e.g., new electrode placements), use:

```
data/s_MM_DD_YY_1
data/s_MM_DD_YY_2
```

Name it anything inside `data/` if consistent.

---

### 6. Train the Model

Open the Jupyter notebook:

```bash
jupyter notebook train_model.ipynb
```

This notebook will:
- Load your recorded data
- Preprocess signals
- Train and save a machine learning model

---

### 7. Run Real-Time Inference

In `hand_prediction_server.py`, set the path to your trained model:

```python
MODEL_DIR = "models/your_model_directory"
```

Stop the existing `hand_client.py` terminal (from `hand_prompt.sh`)  
Then run:

```bash
sh hand_prediction.sh
```

This starts real-time prediction using live EMG/EEG signals.

---

## 🧩 Additional Features

- ✅ Control prosthetic hand via keyboard or webcam (Instructions coming soon)

https://github.com/user-attachments/assets/08382008-1f0d-4633-aaa4-c096e12325b7

- 📉 Built-in filters (EMA, SMA, IIR) in `exg/`

---

## ✅ TODO

- [ ] Make each non-gui server/client threadable
- [ ] Make each gui server/client a QT Widget
- [ ] Create a unified launcher for all services
- [ ] Combine training and real-time inference with mode toggle

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
