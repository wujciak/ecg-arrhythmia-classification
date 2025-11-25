import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential, load_model
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense, Dropout, LSTM
from keras.api.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st


def preprocess_data(file_path):

    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    close_data = df[['Price']]
    return df, close_data


def scale_data(train, test):

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    return train_scaled, test_scaled, scaler


def create_dataset(dataset, time_step):

    dataX, dataY = [], []
    for i in range(len(dataset) - 2 * time_step - 1):
        dataX.append(dataset[i:i + time_step, 0])
        dataY.append(dataset[i + time_step:i + 2 * time_step, 0])
    return np.array(dataX), np.array(dataY)


def reshape_data(X):

    return X.reshape(X.shape[0], X.shape[1], 1)


def build_model(time_step, neurons, dropout_rate):

    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=(time_step, 1), activation='tanh',
             kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        LSTM(neurons * 2, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        LSTM(neurons, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu'),
        Dropout(dropout_rate),
        Dense(time_step, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])
    return model


def plot_loss(history, current_time):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history.history['loss'], 'r', label='Strata treningowa')
    ax.plot(history.history['val_loss'], 'b', label='Strata walidacyjna')

    ax.set_title(f'Strata treningowa i walidacyjna\n{current_time}')
    ax.set_xlabel('Epoki')
    ax.set_ylabel('Strata')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def plot_predictions(df, train_predict, test_predict, train_dates, test_dates, close_data, current_time):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['date'], close_data, label='Cena rzeczywista', color='blue', linewidth=1)
    ax.plot(train_dates, train_predict[:, -1], label='Prognoza treningowa', color='green', linewidth=1.5)
    ax.plot(test_dates, test_predict[:, -1], label='Prognoza testowa', color='orange', linewidth=1.5)

    ax.set_title(f'Ceny vs. Prognozy\n{current_time}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena [USD]')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def predict_future(model, last_data, time_step, days_to_predict, scaler):

    future_predictions = []
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

    current_input = last_data
    for _ in range(days_to_predict):
        next_pred = model.predict(current_input, verbose=0)[0]

        next_pred_rescaled = scaler.inverse_transform(next_pred.reshape(-1, 1))[0][0]
        future_predictions.append(next_pred_rescaled)

        next_pred_scaled = scaler.transform([[next_pred_rescaled]])
        current_input = np.append(current_input[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)

    return np.array(future_predictions), future_dates


def plot_future_predictions(df, historical_snippet, future_dates, future_predictions, time_step, current_time):

    trimmed_df = df.iloc[-2 * time_step:]
    trimmed_dates = trimmed_df['date']
    trimmed_close = trimmed_df['Price']

    historical_snippet = historical_snippet.to_numpy().flatten()

    future_predictions, future_dates = predict_future(model, last_data, time_step, days_to_predict, scaler)
    future_predictions = future_predictions.flatten()

    combined_dates = pd.concat(
        [pd.Series(df['date'].iloc[-time_step:]).reset_index(drop=True), pd.Series(future_dates)], ignore_index=True)

    combined_predictions = np.concatenate((historical_snippet, future_predictions))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].plot(future_dates, future_predictions, label='Przyszłe predykcje', color='red')
    axes[0].fill_between(future_dates, future_predictions - 5, future_predictions + 5, color='red', alpha=0.1)

    axes[0].set_title(f'Predykcje przyszłych cen\n{current_time}', fontsize=12)
    axes[0].set_xlabel('Data', fontsize=10)
    axes[0].set_ylabel('Cena [USD]', fontsize=10)
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(combined_dates[:time_step], historical_snippet, label='Ceny historyczne', color='blue', linewidth=2)
    axes[1].plot(combined_dates[time_step - 1:], combined_predictions[time_step - 1:], label='Przyszłe predykcje',
                 color='red', linewidth=2)
    axes[1].fill_between(future_dates,
                         future_predictions - 5,
                         future_predictions + 5)
    axes[1].set_title(f'Przyszłe predykcje z danymi historycznymi\n{current_time}')
    axes[1].set_xlabel('Data')
    axes[1].set_ylabel('Cena [USD]')
    axes[1].grid()
    axes[1].legend()

    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(combined_dates[:time_step], historical_snippet, label='Ceny historyczne', color='blue', linewidth=2)
    ax.plot(combined_dates[time_step - 1:], combined_predictions[time_step - 1:], label='Przyszłe predykcje',
            color='red', linewidth=2)
    ax.fill_between(future_dates,
                    future_predictions - 5,
                    future_predictions + 5)
    ax.set_title(f'Przyszłe predykcje z danymi historycznymi\n{current_time}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena [USD]')
    ax.legend()
    st.pyplot(fig)


def evaluate_model(y_true, y_pred, dataset_type="Test"):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.write(f"Ewaluacja danych z zestawu {dataset_type}:")
    st.write(f"Średni błąd bezwzględny (MAE): {mae:.4f}")
    st.write(f"Pierwiastek średniego błędu kwadratowego (RMSE): {rmse:.4f}")
    st.write(f"Współczynnik determinacji (R²): {r2:.4f}")


def plot_error_distribution(y_true, y_pred, dataset_type="Test"):

    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors.flatten(), bins=50, color='purple', alpha=0.7)
    ax.set_title(f"Rozkład błędów (Dane {dataset_type})")
    ax.set_xlabel("Błąd")
    ax.set_ylabel("Częstotliwość")
    ax.grid(True)
    st.pyplot(fig)


# Main Workflow
current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

file_path = 'eth_data.csv'
df, close_data = preprocess_data(file_path)

train_size = int(len(close_data) * 0.80)
train_data, test_data = close_data[:train_size], close_data[train_size:]

train_scaled, test_scaled, scaler = scale_data(train_data, test_data)

time_step = 30
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)
X_train, X_test = reshape_data(X_train), reshape_data(X_test)

model_path = 'trained_eth_lstm.keras'
if os.path.exists(model_path):
    print("Trained model exists. Loading the model...")
    model = load_model(model_path)
else:
    print("Model does not exist. Training the model...")
    model = build_model(time_step, neurons=50, dropout_rate=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                                   restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=16, callbacks=[early_stopping])
    model.save(model_path)
    plot_loss(history, current_time)

train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print("\nModel Evaluation:")

train_true = scaler.inverse_transform(y_train)
test_true = scaler.inverse_transform(y_test)

plot_error_distribution(train_true, train_predict, dataset_type="Treningowe")
plot_error_distribution(test_true, test_predict, dataset_type="Testowe")

evaluate_model(train_true, train_predict, dataset_type="Treningowe")
evaluate_model(test_true, test_predict, dataset_type="Testowe")

train_dates = df['date'].iloc[time_step:time_step + len(train_predict)]
test_dates = df['date'].iloc[len(train_predict) + 2 * time_step + time_step:
                             len(train_predict) + 2 * time_step + time_step + len(test_predict)]
plot_predictions(df, train_predict, test_predict, train_dates, test_dates, close_data, current_time)

last_data = test_scaled[-time_step:].reshape(1, time_step, 1)

days_to_predict = 60
future_predictions = predict_future(model, last_data, time_step, days_to_predict, scaler)

last_date = df['date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)

plot_future_predictions(df, close_data[-time_step:], future_dates, future_predictions, time_step, current_time)
