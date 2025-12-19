import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
import os


def load_ecg_data(data_path):
    df = pd.read_csv(data_path)
    return df


def prepare_sequences(X, y, sequence_length=100):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def build_model(input_shape, num_classes, neurons=64, dropout_rate=0.3):
    
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def main():
    # Load data
    data_path = os.path.join('datasets', 'do-ustalenia')
    df = load_ecg_data(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    # Encode labels (normal=0, abnormal=1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare sequences
    sequence_length = 50
    X_seq, y_seq = prepare_sequences(X_scaled, y_encoded, sequence_length)
    y_cat = to_categorical(y_seq, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_cat, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Save model
    model.save('models\lstm\ecg_lstm_model.h5')
    print("Model saved")


if __name__ == "__main__":
    main()