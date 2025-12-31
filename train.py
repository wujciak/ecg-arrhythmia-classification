import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import os
from pathlib import Path
from visualize import plot_training_history


def load_patient_data(data_dir, patient_id):
    data_dir = Path(data_dir)
    ekg_file = data_dir / f"{patient_id}_ekg.csv"
    ann_file = data_dir / f"{patient_id}_annotations_1.csv"
    
    ekg_df = pd.read_csv(ekg_file)
    ann_df = pd.read_csv(ann_file)
    
    return ekg_df, ann_df


def extract_heartbeats(ekg_df, ann_df, window_before=90, window_after=90):
    segments = []
    labels = []
    
    signal_channels = ["MLII", "V5"]
    
    for _, row in ann_df.iterrows():
        idx = int(row["index"])
        symbol = row["annotation_symbol"]
        
        start = idx - window_before
        end = idx + window_after
        
        if start < 0 or end >= len(ekg_df):
            continue
        
        segment = ekg_df.loc[start:end-1, signal_channels].values
        segments.append(segment)
        labels.append(symbol)
    
    return np.array(segments), np.array(labels)


def load_multiple_patients(data_dir, patient_ids):
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


def build_lstm_model(input_shape, num_classes, neurons=64, dropout_rate=0.3):
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape, activation="tanh", kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        LSTM(neurons * 2, return_sequences=True, activation="tanh", kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        LSTM(neurons, activation="tanh", kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(neurons, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model


def build_hybrid_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(units=200, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model


def main():
    model_type = "lstm"  # do wyboru
    data_dir = "datasets/mit-bih-arrhythmia"
    patient_ids = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121
        ]
    
    print("\nLoading data")
    X, y = load_multiple_patients(data_dir, patient_ids)
    
    print(f"\nData shape: {X.shape}")
    print(f"Total heartbeats: {len(y):,}")
    
    y_grouped = group_to_binary(y)
    valid_mask = y_grouped != "Non-beat"
    X = X[valid_mask]
    y_grouped = y_grouped[valid_mask]
    
    print("\nClass distribution:")
    unique, counts = np.unique(y_grouped, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count:,} ({100*count/len(y_grouped):.1f}%)")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_grouped)
    num_classes = len(np.unique(y_encoded))
    y_cat = to_categorical(y_encoded, num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_train = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    print(f"\nTrain: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
    
    print(f"\n{'='*30}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*30}")
    
    if model_type == "lstm":
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
    elif model_type == "hybrid":
        model = build_hybrid_model((X_train.shape[1], X_train.shape[2]), num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'lstm' or 'hybrid'")

    print("\nModel Architecture:")
    model.summary()
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    print("\nTraining start")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    print(f"Classes: {le.classes_}")
    
    model_dir = f"models/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ecg_{model_type}_model.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\nGenerating training plots...")
    plot_training_history(history, model_type)
    print(f"Training plots saved to plots/training/{model_type}_history.png")


if __name__ == "__main__":
    main()
