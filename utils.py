import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path


def load_patient_data(data_dir, patient_id):
    """Load ECG and annotation data for a specific patient."""
    data_dir = Path(data_dir)
    ekg_file = data_dir / f"{patient_id}_ekg.csv"
    ann_file = data_dir / f"{patient_id}_annotations_1.csv"
    
    ekg_df = pd.read_csv(ekg_file)
    ann_df = pd.read_csv(ann_file)
    
    return ekg_df, ann_df


def extract_heartbeats(ekg_df, ann_df, window_before=90, window_after=90):
    """Extract heartbeat windows from ECG signal centered on annotations."""
    segments = []
    labels = []
    
    exclude_cols = ['Unnamed: 0', 'symbol', 'index', 'time']
    signal_channels = [col for col in ekg_df.columns if col not in exclude_cols]
    
    first_lead = signal_channels[0]
    second_lead = signal_channels[1] if len(signal_channels) > 1 else signal_channels[0]
    
    for _, row in ann_df.iterrows():
        idx = int(row["index"])
        symbol = row["annotation_symbol"]
        
        start = idx - window_before
        end = idx + window_after
        
        if start < 0 or end >= len(ekg_df):
            continue
        
        segment = ekg_df.loc[start:end-1, [first_lead, second_lead]].values
        segments.append(segment)
        labels.append(symbol)
    
    return np.array(segments), np.array(labels)


def load_multiple_patients(data_dir, patient_ids):
    """Load and combine data from multiple patients."""
    all_segments = []
    all_labels = []
    
    for patient_id in patient_ids:
        try:
            ekg_df, ann_df = load_patient_data(data_dir, patient_id)
            segments, labels = extract_heartbeats(ekg_df, ann_df)
            all_segments.append(segments)
            all_labels.append(labels)
            print(f"Loaded patient {patient_id}: {len(segments)} heartbeats")
        except Exception as e:
            print(f"Error loading patient {patient_id}: {e}")
    
    return np.concatenate(all_segments), np.concatenate(all_labels)


def group_to_binary(labels):
    """Group arrhythmia annotations into binary classes: Normal, Abnormal, Non-beat."""
    mapping = {
        "N": "Normal", 
        "L": "Normal", 
        "R": "Normal", 
        ".": "Normal",
        
        "A": "Abnormal", 
        "a": "Abnormal", 
        "J": "Abnormal", 
        "S": "Abnormal",
        "V": "Abnormal", 
        "E": "Abnormal", 
        "F": "Abnormal", 
        "/": "Abnormal",
        "f": "Abnormal", 
        "Q": "Abnormal",
        
        "+": "Non-beat", 
        "~": "Non-beat", 
        "|": "Non-beat"
    }
    return np.array([mapping.get(label, "Abnormal") for label in labels])


def plot_training_history(history, model_name='model'):
    """Plot and save training history (accuracy and loss)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # loss plot
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/training', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'plots/training/{model_name}_history_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Training history saved to {filename}")
