import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_patient_data(data_dir, patient_id):
    data_dir = Path(data_dir)
    ekg_file = data_dir / f"{patient_id}_ekg.csv"
    ann_file = data_dir / f"{patient_id}_annotations_1.csv"
    
    ekg_df = pd.read_csv(ekg_file)
    ann_df = pd.read_csv(ann_file)
    
    return ekg_df, ann_df


def plot_ecg_signal(ekg_df, patient_id, num_samples=3600):
    plt.figure(figsize=(16, 6))
    time = np.arange(num_samples) / 360  # Convert to seconds
    plt.plot(time, ekg_df["MLII"].values[:num_samples], label="MLII", alpha=0.8, linewidth=1)
    plt.plot(time, ekg_df["V5"].values[:num_samples], label="V5", alpha=0.8, linewidth=1)
    
    plt.title(f"ECG Signal - Patient {patient_id} (First 10 seconds)", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("plots/signals", exist_ok=True)
    plt.savefig(f"plots/signals/ecg_signal_patient_{patient_id}.png", dpi=150)
    plt.show()
    print(f"Signal plot saved to plots/signals/ecg_signal_patient_{patient_id}.png")


def plot_annotated_signal(ekg_df, ann_df, patient_id, start_sample=0, duration_sec=10):
    num_samples = int(duration_sec * 360)  # 360 Hz sampling rate
    end_sample = start_sample + num_samples
    
    plt.figure(figsize=(16, 6))
    
    time = np.arange(start_sample, end_sample) / 360
    signal = ekg_df["MLII"].values[start_sample:end_sample]
    plt.plot(time, signal, "b-", alpha=0.7, linewidth=1.5, label="MLII")
    
    ann_in_range = ann_df[(ann_df["index"] >= start_sample) & (ann_df["index"] < end_sample)]
    
    for _, row in ann_in_range.iterrows():
        idx = row["index"]
        symbol = row["annotation_symbol"]
        time_point = idx / 360
        signal_value = ekg_df.loc[idx, "MLII"]
        
        color = "green" if symbol == 'N' else "red"
        plt.plot(time_point, signal_value, 'o', color=color, markersize=8, alpha=0.7)
        plt.text(time_point, signal_value + 0.1, symbol, fontsize=10, ha="center", fontweight="bold")
    
    plt.title(f"ECG with Annotations - Patient {patient_id}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("plots/annotated", exist_ok=True)
    plt.savefig(f"plots/annotated/annotated_signal_patient_{patient_id}.png", dpi=150)
    plt.show()
    print(f"Annotated signal saved to plots/annotated/annotated_signal_patient_{patient_id}.png")


def print_data_info(ekg_df, ann_df, patient_id):
    print("="*30)
    print(f"Patient {patient_id} - Data Info")
    print("="*30)
    
    print(f"\nEKG Signal:")
    print(f"  Shape: {ekg_df.shape}")
    print(f"  Channels: {["MLII:", "V5"]}")
    print(f"  Duration: {len(ekg_df) / 360 / 60:.1f} minutes")
    print(f"  Sampling rate: 360 Hz")
    
    print(f"\nAnnotations:")
    print(f"  Total heartbeats: {len(ann_df)}")
    print(f"  Unique symbols: {ann_df["annotation_symbol"].nunique()}")
    
    print(f"\nAnnotation distribution:")
    for symbol, count in ann_df["annotation_symbol"].value_counts().items():
        print(f"  {symbol}: {count} ({100*count/len(ann_df):.1f}%)")


def plot_class_distribution(ann_df, patient_id=None):
    plt.figure(figsize=(10, 6))
    
    counts = ann_df["annotation_symbol"].value_counts()
    colors = ["green" if symbol == 'N' else "red" for symbol in counts.index]
    
    counts.plot(kind="bar", color=colors, alpha=0.7, edgecolor="black")
    
    title = f"Heartbeat Annotation Distribution"
    if patient_id:
        title += f" - Patient {patient_id}"
    
    plt.title(title, fontsize=14, fontweight="bold:")
    plt.xlabel("Annotation Symbol")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("plots/distributions", exist_ok=True)
    filename = f"plots/distributions/class_distribution_patient_{patient_id}.png" if patient_id else "plots/distributions/class_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Distribution plot saved to {filename}")


def plot_training_history(history, model_name='model'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots/training', exist_ok=True)
    plt.savefig(f'plots/training/{model_name}_history.png', dpi=150)
    plt.show()
    print(f"Training history saved to plots/training/{model_name}_history.png")


def plot_heartbeat_samples(ekg_df, ann_df, num_samples=5, window=90):
    symbols = ann_df['annotation_symbol'].value_counts().head(5).index
    
    fig, axes = plt.subplots(len(symbols), num_samples, figsize=(15, 3*len(symbols)))
    
    for i, symbol in enumerate(symbols):
        symbol_anns = ann_df[ann_df['annotation_symbol'] == symbol]
        sample_anns = symbol_anns.sample(min(num_samples, len(symbol_anns)))
        
        for j, (_, row) in enumerate(sample_anns.iterrows()):
            idx = int(row['index'])
            start = max(0, idx - window)
            end = min(len(ekg_df), idx + window)
            
            segment = ekg_df['MLII'].values[start:end]
            
            ax = axes[i, j] if len(symbols) > 1 else axes[j]
            ax.plot(segment, linewidth=1.5)
            
            if j == 0:
                ax.set_ylabel(f'{symbol}', fontsize=12, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Heartbeats by Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('plots/samples', exist_ok=True)
    plt.savefig('plots/samples/heartbeat_samples.png', dpi=150)
    plt.show()
    print("Heartbeat samples saved to plots/samples/heartbeat_samples.png")


def main():
    data_dir = 'datasets/mit-bih-arrhythmia'
    patient_id = 100  # do zmiany w zależności od pacjenta
    
    print(f"Loading data for patient {patient_id}\n")
    ekg_df, ann_df = load_patient_data(data_dir, patient_id)
    
    print_data_info(ekg_df, ann_df, patient_id)
    
    print("\nVisualizations:")
    plot_ecg_signal(ekg_df, patient_id, num_samples=3600)
    plot_annotated_signal(ekg_df, ann_df, patient_id, start_sample=0, duration_sec=10)
    plot_class_distribution(ann_df, patient_id)
    plot_heartbeat_samples(ekg_df, ann_df, num_samples=5, window=90)
    print("All plots saved to plots/ directory")


if __name__ == "__main__":
    main()
