import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.metrics import Precision, Recall

import os
from utils import plot_training_history, load_multiple_patients, group_to_binary


def build_lstm_model(input_shape, num_classes, neurons=64, dropout_rate=0.3):
    """Build a multi-layer LSTM model for ECG arrhythmia classification."""
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
    
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=[
            "accuracy",
            Precision(class_id=0, name='abnormal_precision'),
            Recall(class_id=0, name='abnormal_recall')
        ]
    )
    
    return model


def build_hybrid_model(input_shape, num_classes):
    """Build a hybrid CNN-LSTM model combining convolutional feature extraction with temporal modeling."""
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(units=200, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=[
            "accuracy",
            Precision(class_id=0, name='abnormal_precision'),
            Recall(class_id=0, name='abnormal_recall')
        ]
    )
    
    return model


def main():
    """Main training pipeline: load data, preprocess, train model, and evaluate results."""
    model_type = "lstm"  # lstm or hybrid
    data_dir = "datasets/mit-bih"    
    patient_ids = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
        111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
        200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 
        217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
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
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    
    print(f"\nClass Weights:")
    for i, weight in class_weights.items():
        print(f"  {le.classes_[i]}: {weight:.2f}")
    
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
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Abnormal Precision: {results[2]*100:.2f}%")
    print(f"Test Abnormal Recall (Sensitivity): {results[3]*100:.2f}%")
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    bal_acc = balanced_accuracy_score(y_test_classes, y_pred_classes)
    print(f"\nTest Balanced Accuracy: {bal_acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(f"{'':>15} {'Pred Abnormal':>15} {'Pred Normal':>15}")
    for i, label in enumerate(le.classes_):
        print(f"{f'True {label}':>15} {cm[i][0]:>15} {cm[i][1]:>15}")
    
    print("\nPer-Class Metrics:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_, digits=4))
    
    if len(le.classes_) == 2:
        abnormal_idx = np.where(le.classes_ == 'Abnormal')[0][0]
        normal_idx = np.where(le.classes_ == 'Normal')[0][0]
        
        tn = cm[normal_idx][normal_idx]
        fp = cm[normal_idx][abnormal_idx]
        fn = cm[abnormal_idx][normal_idx]
        tp = cm[abnormal_idx][abnormal_idx]
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nMedical Metrics:")
        print(f"  Sensitivity (Abnormal Detection): {sensitivity*100:.2f}%")
        print(f"  Specificity (Normal Detection): {specificity*100:.2f}%")
    
    print(f"\nClasses: {le.classes_}")
    
    model_dir = f"models/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ecg_{model_type}_model.keras"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    plot_training_history(history, model_type)


if __name__ == "__main__":
    main()
