import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout
import streamlit as st
from keras.api.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import time


def load_and_preprocess_data(file_path, time_step = 72):

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.dropna(subset=['Price'], inplace=True)

    df.reset_index(drop=True, inplace=True)
    close_data = df['Price'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_data)

    dataX, dataY = [], []
    for i in range(time_step, len(scaled_prices)):
        dataX.append(scaled_prices[i - time_step:i, 0])
        dataY.append(scaled_prices[i, 0])

    dataX, dataY = np.array(dataX), np.array(dataY)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    return dataX, dataY, scaler, df, close_data


def build_model(input_shape):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(units=200, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])
    return history


def plot_predictions(df, train_predict, test_predict, train_dates, test_dates, close_data, current_time):

    train_predict = train_predict.flatten()
    test_predict = test_predict.flatten()

    test_dates = test_dates[:len(test_predict)]
    train_dates = train_dates[:len(train_predict)]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['date'], close_data, label='Cena rzeczywista', color='blue', linewidth=1)
    ax.plot(train_dates, train_predict, label='Prognoza treningowa', color='green', linewidth=1.5)
    ax.plot(test_dates, test_predict, label='Prognoza testowa', color='orange', linewidth=1.5)
    ax.set_title(f'Ceny vs. Prognozy\n{current_time}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena [USD]')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def evaluate_model(model, dataX, dataY, scaler, dataset_name):
    predictions = model.predict(dataX)
    y_pred = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(dataY.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    st.write(f"Ewaluacja danych z zestawu {dataset_name}:")
    st.write(f"Średni błąd bezwzględny (MAE): {mae:.4f}")
    st.write(f"Pierwiastek średniego błędu kwadratowego (RMSE): {rmse:.4f}")
    st.write(f"Współczynnik determinacji (R²): {r2:.4f}")

    return y_pred

def predict_future_prices(model, last_data, scaler, days_ahead=70):

    future_predictions = []
    for _ in range(days_ahead):
        prediction = model.predict(last_data)
        future_predictions.append(scaler.inverse_transform(prediction)[0, 0])
        last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    return future_predictions


def plot_future_prices(y_test, future_prices, scaler, days_ahead=30, context_length=100):

    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    context_data = y_test_rescaled[-context_length:]
    historical_indices = np.arange(len(y_test) - context_length, len(y_test))
    future_indices = np.arange(len(y_test), len(y_test) + days_ahead)

    aligned_future_prices = [context_data[-1][0]] + future_prices

    aligned_future_indices = np.arange(len(y_test) - 1, len(y_test) + days_ahead)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(historical_indices, context_data, label="Ceny historyczne", color="blue")
    ax.plot(aligned_future_indices, aligned_future_prices, label="Przyszłe predykcje", linestyle="-", color="red")
    ax.set_title('Przyszłe predykcje z danymi historycznymi')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena [USD]')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def plot_error_distribution(dataX, dataY, scaler, dataset_type="Test"):

    predictions = model.predict(dataX)
    y_pred = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(dataY.reshape(-1, 1))
    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors.flatten(), bins=50, color='purple', alpha=0.7)
    ax.set_title(f"Rozkład błędów ({dataset_type} Dane)")
    ax.set_xlabel("Błąd")
    ax.set_ylabel("Częstotliwość")
    ax.grid(True)
    st.pyplot(fig)


# Main Workflow
file_path = 'eth_data.csv'
time_step = 72
dataX, dataY, scaler, df, close_data = load_and_preprocess_data(file_path, time_step)

dates = df['date']
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, shuffle=False)
train_dates = dates[:len(y_train)]
test_dates = dates[len(y_train):]

model_path = 'eth_hybrid.keras'
if os.path.exists(model_path):
    print("Loading trained model...")
    model = load_model(model_path)
else:
    print("Training a new model...")
    model = build_model(input_shape=(X_train.shape[1], 1))
    train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    model.save(model_path)

print("Evaluating training data...")
train_predict = evaluate_model(model, X_train, y_train, scaler, "Treningowe")
plot_error_distribution(X_train, y_train, scaler, "Train")

print("Evaluating testing data...")
test_predict = evaluate_model(model, X_test, y_test, scaler, "Testowe")
plot_error_distribution(X_test, y_test, scaler, "Test")

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
plot_predictions(df, train_predict, test_predict, train_dates, test_dates, close_data, current_time)

last_data = X_test[-1].reshape(1, X_test.shape[1], 1)
future_prices = predict_future_prices(model, last_data, scaler, days_ahead=30)

plot_future_prices(y_test, future_prices, scaler, days_ahead=30, context_length=100)
days_ahead = 30

print(f"Future predictions for the next {days_ahead} days:")
print(future_prices)





