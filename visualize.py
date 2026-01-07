import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def get_signal_channels(ekg_df):
    """Extract ECG signal channel names, excluding metadata columns."""
    exclude_cols = ['Unnamed: 0', 'symbol', 'index', 'time']
    channels = [col for col in ekg_df.columns if col not in exclude_cols]
    return channels


def load_patient_data(data_dir, patient_id):
    """Load ECG and annotation data for a specific patient."""
    data_dir = Path(data_dir)
    ekg_file = data_dir / f"{patient_id}_ekg.csv"
    ann_file = data_dir / f"{patient_id}_annotations_1.csv"
    
    ekg_df = pd.read_csv(ekg_file)
    ann_df = pd.read_csv(ann_file)
    
    return ekg_df, ann_df


def plot_ecg_signal(ekg_df, patient_id, num_samples=3600):
    """Plot and save raw ECG signal for the first 10 seconds."""
    signal_channels = get_signal_channels(ekg_df)
    
    plt.figure(figsize=(16, 6))
    time = np.arange(num_samples) / 360  # convert to seconds
    
    for channel in signal_channels[:2]:
        plt.plot(time, ekg_df[channel].values[:num_samples], label=channel, alpha=0.8, linewidth=1)
    
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
    """Plot ECG signal with annotation markers colored by beat type."""
    signal_channels = get_signal_channels(ekg_df)
    primary_channel = signal_channels[0]
    
    num_samples = int(duration_sec * 360)  # 360 Hz sampling rate
    end_sample = start_sample + num_samples
    
    plt.figure(figsize=(16, 6))
    
    time = np.arange(start_sample, end_sample) / 360
    signal = ekg_df[primary_channel].values[start_sample:end_sample]
    plt.plot(time, signal, "b-", alpha=0.7, linewidth=1.5, label=primary_channel)
    
    ann_in_range = ann_df[(ann_df["index"] >= start_sample) & (ann_df["index"] < end_sample)]
    
    for _, row in ann_in_range.iterrows():
        idx = row["index"]
        symbol = row["annotation_symbol"]
        time_point = idx / 360
        signal_value = ekg_df.loc[idx, primary_channel]
        
        if symbol in ['N', 'L', 'R', '.']:
            color = "green"
        elif symbol in ['+', '~', '|']:
            color = "grey"
        else:
            color = "red"
        
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
    """Print summary statistics about ECG recording and annotations."""
    signal_channels = get_signal_channels(ekg_df)
    
    print("="*30)
    print(f"Patient {patient_id} - Data Info")
    print("="*30)
    
    print(f"\nEKG Signal:")
    print(f"  Shape: {ekg_df.shape}")
    print(f"  Channels: {signal_channels}")
    print(f"  Duration: {len(ekg_df) / 360 / 60:.1f} minutes")
    print(f"  Sampling rate: 360 Hz")
    
    print(f"\nAnnotations:")
    print(f"  Total heartbeats: {len(ann_df)}")
    print(f"  Unique symbols: {ann_df["annotation_symbol"].nunique()}")
    
    print(f"\nAnnotation distribution:")
    for symbol, count in ann_df["annotation_symbol"].value_counts().items():
        print(f"  {symbol}: {count} ({100*count/len(ann_df):.1f}%)")


def plot_class_distribution(ann_df, patient_id=None):
    """Plot bar chart showing distribution of heartbeat annotation types."""
    plt.figure(figsize=(10, 6))
    
    counts = ann_df["annotation_symbol"].value_counts()
    colors = ["green" if symbol in ['N', 'L', 'R', '.'] else "grey" if symbol in ['+', '~', '|'] else "red" for symbol in counts.index]
    
    counts.plot(kind="bar", color=colors, alpha=0.7, edgecolor="black")
    
    title = f"Heartbeat Annotation Distribution"
    if patient_id:
        title += f" - Patient {patient_id}"
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Annotation Symbol")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("plots/distributions", exist_ok=True)
    filename = f"plots/distributions/class_distribution_patient_{patient_id}.png" if patient_id else "plots/distributions/class_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Distribution plot saved to {filename}")


def plot_heartbeat_samples(ekg_df, ann_df, num_samples=5, window=90):
    """Plot sample heartbeat waveforms for each annotation type."""
    signal_channels = get_signal_channels(ekg_df)
    primary_channel = signal_channels[0]
    
    symbols = ann_df['annotation_symbol'].value_counts().head(5).index
    
    fig, axes = plt.subplots(len(symbols), num_samples, figsize=(15, 3*len(symbols)))
    
    for i, symbol in enumerate(symbols):
        symbol_anns = ann_df[ann_df['annotation_symbol'] == symbol]
        sample_anns = symbol_anns.sample(min(num_samples, len(symbol_anns)))
        
        for j, (_, row) in enumerate(sample_anns.iterrows()):
            idx = int(row['index'])
            start = max(0, idx - window)
            end = min(len(ekg_df), idx + window)
            
            segment = ekg_df[primary_channel].values[start:end]
            
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
    """Generate all visualizations for a single patient's ECG data."""
    data_dir = 'datasets/mit-bih'
    patient_id = 200  # do zmiany w zależności od pacjenta
    
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
