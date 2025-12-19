import pandas as pd
import matplotlib.pyplot as plt
import os


def load_ecg_data(data_path):
    df = pd.read_csv(data_path)
    return df


def plot_ecg_signal(df, num_samples=1000):
    plt.figure(figsize=(14, 5))
    for col in df.columns[:-1][:2]:  # First 2 leads
        plt.plot(df[col].values[:num_samples], label=col, alpha=0.7)
    plt.title('ECG Signal Sample')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('data_plots/ecg_signal.png', dpi=150)
    plt.show()


def print_data_info(df):
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nClass distribution:\n{df.iloc[:, -1].value_counts()}")


def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    label_col = df.columns[-1]
    df[label_col].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('ECG Classification Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data_plots/class_distribution.png', dpi=150)
    plt.show()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()


def main():
    data_path = os.path.join('datasets', 'idk', 'idk')
    df = load_ecg_data(data_path)
    print_data_info(df)
    plot_ecg_signal(df)
    plot_class_distribution(df)


if __name__ == "__main__":
    main()