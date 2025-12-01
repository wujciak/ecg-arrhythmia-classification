# ECG Signal Classification using LSTM

## Project Description

This project uses LSTM (Long Short-Term Memory) neural networks to classify ECG (Electrocardiogram) signals as **Normal** or **Abnormal**. The model is trained on annotated ECG data and can detect various types of cardiac arrhythmias.

### Classification Categories:
- **Normal beats**: N (Normal), L (Left bundle branch block), R (Right bundle branch block), e (Atrial escape), j (Nodal escape)
- **Abnormal beats**: A (Atrial premature), V (Ventricular premature), S (Supraventricular premature), F (Fusion), and others

## Features

1. **Data Visualization**:
   - Browse ECG recordings with annotations
   - View Normal (green) and Abnormal (red) beat markers
   - Explore MLII and V5 leads

2. **Model Training**:
   - LSTM-based binary classifier
   - Configurable hyperparameters (epochs, batch size, neurons, dropout)
   - Early stopping and model checkpointing
   - Training progress visualization

3. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-Score metrics
   - Confusion matrix visualization
   - ROC curve with AUC score
   - Sample predictions display

4. **Single Record Prediction**:
   - Classify all beats in a specific ECG recording
   - Per-beat prediction with confidence scores

## Project Structure

- **Main files:**
  - `ecg_lstm_classifier.py` – LSTM classifier implementation
  - `app.py` – Streamlit web interface
  - `requirements.txt` – Project dependencies

- **Data directory (`data/`):**
  - `{record}_ekg.csv` – ECG signal data (MLII, V5 leads)
  - `{record}_annotations_1.csv` – Beat annotations

- **Legacy files (`legacy/`):**
  - Cryptocurrency prediction models (BTC/ETH)

## Requirements

- Python 3.8 or newer
- Installed Python packages:
  - `streamlit`
  - `keras`
  - `tensorflow`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run Training Script Directly
```bash
python ecg_lstm_classifier.py
```

## Model Architecture

The LSTM classifier consists of:
- 3 LSTM layers with batch normalization and dropout
- Dense layers for classification
- Sigmoid activation for binary output (Normal/Abnormal)
- Binary cross-entropy loss function
- Adam optimizer

## Data Format

### ECG Signal File (`{record}_ekg.csv`)
| Column | Description |
|--------|-------------|
| Index  | Sample number |
| MLII   | Lead II amplitude (mV) |
| V5     | Lead V5 amplitude (mV) |
| symbol | Beat annotation (optional) |

### Annotation File (`{record}_annotations_1.csv`)
| Column | Description |
|--------|-------------|
| index  | Sample index of the beat |
| annotation_symbol | Beat type (N, A, V, etc.) |
  - `numpy`
  - `matplotlib`
  - `os`
  - `datetime`
-

## Instalacja

1. Sklonuj projekt lub pobierz pliki.
2. Zainstaluj wymagane pakiety.
3. Uruchom za pomocą komendy: streamlit run interface.py

