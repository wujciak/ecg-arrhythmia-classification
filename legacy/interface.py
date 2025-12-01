import streamlit as st
import pandas as pd
import os
from datetime import datetime
from keras.api.models import load_model
from keras.api.callbacks import EarlyStopping

from eth.ETH_lstm import preprocess_data, scale_data, create_dataset, reshape_data, build_model, plot_loss, \
    plot_predictions, evaluate_model, plot_error_distribution, predict_future, plot_future_predictions
from btc.BTC_lstm import preprocess_data as preprocess_data_b, \
    scale_data as scale_data_b, create_dataset as create_dataset_b, \
    plot_loss as plot_loss_b, plot_predictions as plot_predictions_b, \
    evaluate_model as evaluate_model_b, plot_error_distribution as plot_error_distribution_b, \
    predict_future as predict_future_b, plot_future_predictions as plot_future_predictions_b
from eth.eth_visual import visualize_eth
from btc.btc_visual import visualize_btc

from eth.ETH_hybrid import load_and_preprocess_data as load_eth_data, build_model as build_eth_hybrid_model, \
    train_model as train_eth_model, predict_future_prices as predict_eth_future_prices, \
    plot_predictions as plot_eth_predictions, plot_future_prices as plot_eth_future_prices, \
    evaluate_model as evaluate_eth_model
from btc.BTC_hybrid import load_and_preprocess_data as load_btc_data, build_model as build_btc_hybrid_model, \
    train_model as train_btc_model, predict_future_prices as predict_btc_future_prices, \
    plot_predictions as plot_btc_predictions, plot_future_prices as plot_btc_future_prices, \
    evaluate_model as evaluate_btc_model


st.title("Prognozowanie cen kryptowalut i ich wizualizacja")

st.sidebar.title("Opcje")
dataset_option = st.sidebar.selectbox("Wybierz zestaw danych", ["Wybierz", "ETH", "BTC"])
option = st.sidebar.selectbox("Wybierz opcję",
                              ["Wybierz", "Wizualizuj dane", "Trenuj i prognozuj", "Oceń model"])
model_option = st.sidebar.selectbox("Wybierz typ modelu",
                                    ["Wybierz", "LSTM", "LSTM+CNN (Hybrid)"])

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
eth_file_path = 'eth_data.csv'
btc_file_path = 'btc_data.csv'

file_path = eth_file_path if dataset_option == "ETH" else btc_file_path if dataset_option == "BTC" else None

if dataset_option == "ETH":
    st.success("Wybrano zestaw danych Ethereum (ETH).")
elif dataset_option == "BTC":
    st.success("Wybrano zestaw danych Bitcoin (BTC).")
else:
    st.warning("Proszę wybrać zestaw danych, aby kontynuować.")

# Opcja 1: Wizuazlizacja danych
if option == "Wizualizuj dane" and file_path:
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    st.subheader("Pierwsze wiersze zestawu danych:")
    st.write(df.tail())
    st.success("Wizualizacja danych.")

    if dataset_option == "ETH":
        try:
            visualize_eth(file_path)
        except Exception as e:
            st.error(f"Błąd podczas ładowania zestawu danych Ethereum: {e}")
    elif dataset_option == "BTC":
        try:
            visualize_btc(file_path)
        except Exception as e:
            st.error(f"Błąd podczas ładowania zestawu danych Bitcoin: {e}")

# Opcja 2: Trenuj i prognozuj
if option == "Trenuj i prognozuj" and file_path:

    # Krok 1: Wybór modelu i zestawu danych
    if model_option == "LSTM":
        st.subheader("Trening i prognozowanie za pomocą LSTM")

        if dataset_option == "ETH":
            df, close_data = preprocess_data(file_path)
        else:
            df, close_data = preprocess_data_b(file_path)

        train_size = int(len(close_data) * 0.80)
        train_data, test_data = close_data[:train_size], close_data[train_size:]

        if dataset_option == "ETH":
            train_scaled, test_scaled, scaler = scale_data(train_data, test_data)
        else:
            train_scaled, test_scaled, scaler = scale_data_b(train_data, test_data)

        time_step = 30
        if dataset_option == "ETH":
            X_train, y_train = create_dataset(train_scaled, time_step)
            X_test, y_test = create_dataset(test_scaled, time_step)
        else:
            X_train, y_train = create_dataset_b(train_scaled,
                                                time_step)
            X_test, y_test = create_dataset_b(test_scaled, time_step)

        X_train, X_test = reshape_data(X_train), reshape_data(X_test)

        # Krok 2: Załaduj lub trenuj model
        model_path = f'trained_{dataset_option.lower()}_lstm.keras'
        if os.path.exists(model_path):
            st.write(f"Model znaleziony. Ładowanie wcześniej wytrenowanego modelu {dataset_option}...")
            model = load_model(model_path)
        else:
            st.write(f"Nie znaleziono wcześniej wytrenowanego modelu. Trenowanie nowego modelu {dataset_option}...")
            if dataset_option == "ETH":
                model = build_model(time_step, neurons=50, dropout_rate=0.2)
                early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                                               restore_best_weights=True)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    epochs=100, batch_size=16, callbacks=[early_stopping])
            else:
                model = build_model(time_step, neurons=256, dropout_rate=0.3)
                early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                                               restore_best_weights=True)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    epochs=100, batch_size=32, callbacks=[early_stopping])
            model.save(model_path)

            if dataset_option == "ETH":
                plot_loss(history, current_time)  # Rysowanie wykresu straty dla ETH
            else:
                plot_loss_b(history, current_time)  # Rysowanie wykresu straty dla BTC

        # Krok 3: Prognozy
        train_predict = scaler.inverse_transform(model.predict(X_train))
        test_predict = scaler.inverse_transform(model.predict(X_test))
        train_true = scaler.inverse_transform(y_train)
        test_true = scaler.inverse_transform(y_test)

        # Krok 4: Wizualizacja prognoz natychmiast
        st.subheader("Wykresy prognoz")
        train_dates = df['date'].iloc[time_step:time_step + len(train_predict)]
        test_dates = df['date'].iloc[len(train_predict) + 2 * time_step + time_step:
                                     len(train_predict) + 2 * time_step + time_step + len(test_predict)]

        if dataset_option == "ETH":
            plot_predictions(df, train_predict, test_predict, train_dates, test_dates, close_data,
                             current_time)
        else:
            plot_predictions_b(df, train_predict, test_predict, train_dates, test_dates, close_data,
                               current_time)

        # Krok 5: Prognozy na przyszłość
        st.subheader("Prognozy na przyszłość")
        days_to_predict = st.number_input("Wprowadź liczbę dni do prognozy:", min_value=1, max_value=365, value=30)

        if st.button(f"Prognozuj ceny na przyszłość ({dataset_option})"):
            last_data = test_scaled[-time_step:].reshape(1, time_step, 1)
            if dataset_option == "ETH":
                future_predictions, future_dates = predict_future(model, last_data, time_step, days_to_predict,
                                                                  scaler)
                plot_future_predictions(df, close_data[-time_step:], future_dates, future_predictions, time_step,
                                        current_time)
            else:
                future_predictions, future_dates = predict_future_b(model, last_data, time_step, days_to_predict,
                                                                    scaler)
                plot_future_predictions_b(df, close_data[-time_step:], future_dates, future_predictions, time_step,
                                          current_time)

            # Przyszłe prognozy w formie tabeli
            st.subheader("Prognoza przyszłych wartości")
            future_predictions_r = [round(value, 2) for value in future_predictions]
            future_data = pd.DataFrame({
                "Data": future_dates,
                "Prognozowana cena": future_predictions_r
            })
            st.table(future_data)

    elif model_option == "LSTM+CNN (Hybrid)":

        time_step = 72
        if dataset_option == "ETH":
            X, y, scaler, df, close_data = load_eth_data(file_path, time_step)
        else:
            X, y, scaler, df, close_data = load_btc_data(file_path, time_step)

        train_size = int(len(y) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model_path = f'{dataset_option.lower()}_hybrid.keras'
        if os.path.exists(model_path):
            st.write(f"Model znaleziony. Ładowanie wcześniej wytrenowanego modelu {dataset_option}...")
            model = load_model(model_path)
        else:
            st.write(f"Nie znaleziono wcześniej wytrenowanego modelu. Trenowanie nowego modelu {dataset_option}...")
            if dataset_option == "ETH":
                model = build_eth_hybrid_model(input_shape=(X.shape[1], 1))
                train_btc_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
            elif dataset_option == "BTC":
                model = build_btc_hybrid_model(input_shape=(X.shape[1], 1))
                train_eth_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

            model.save(model_path)

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        st.write("Wykresy prognoz:")
        plot_eth_predictions(df, train_predict, test_predict, df['date'][:len(y_train)], df['date'][len(y_train):],
                             close_data, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) \
            if dataset_option == "ETH" \
            else \
            plot_btc_predictions(df, train_predict, test_predict, df['date'][:len(y_train)], df['date'][len(y_train):],
                                 close_data, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        st.subheader("Prognozy na przyszłość")
        future_days = st.number_input("Wprowadź liczbę dni do prognozy:", min_value=1, max_value=365, value=30)

        if st.button(f"Prognozuj ceny na przyszłość ({dataset_option})"):
            last_data = X_test[-1:].reshape(1, X_test.shape[1], 1)
            if dataset_option == "ETH":
                future_prices, future_days = predict_eth_future_prices(model, last_data, scaler, days_ahead=future_days)
                plot_eth_future_prices(y_test, future_prices, scaler, days_ahead=future_days)
            else:
                future_prices, future_days = predict_btc_future_prices(model, last_data, scaler, days_ahead=future_days)
                plot_btc_future_prices(y_test, future_prices, scaler, days_ahead=future_days)

            # Przyszłe prognozy w formie tabeli
            st.subheader("Prognoza przyszłych wartości")
            future_predictions_r = [round(value, 2) for value in future_prices]
            future_data = pd.DataFrame({
                "Data": future_days,
                "Prognozowana cena": future_predictions_r
                })
            st.table(future_data)


# Opcja 3: Ewaluacja modelu
elif option == "Oceń model" and file_path:
    st.subheader("Ocena modelu")

    if model_option == 'LSTM':
        model_path = f'trained_{dataset_option.lower()}_lstm.keras'
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.write(f"Załadowano model dla {dataset_option}.")

            # Przygotowanie danych do oceny
            if dataset_option == "ETH":
                df, close_data = preprocess_data(file_path)
            else:
                df, close_data = preprocess_data_b(file_path)

            train_size = int(len(close_data) * 0.80)
            train_data, test_data = close_data[:train_size], close_data[train_size:]

            if dataset_option == "ETH":
                train_scaled, test_scaled, scaler = scale_data(train_data, test_data)
            else:
                train_scaled, test_scaled, scaler = scale_data_b(train_data, test_data)

            # Tworzenie zestawów danych zgodnych z LSTM
            time_step = 30
            if dataset_option == "ETH":
                X_train, y_train = create_dataset(train_scaled, time_step)
                X_test, y_test = create_dataset(test_scaled, time_step)
            else:
                X_train, y_train = create_dataset_b(train_scaled, time_step)
                X_test, y_test = create_dataset_b(test_scaled, time_step)

            X_train, X_test = reshape_data(X_train), reshape_data(X_test)

            # Prognozy
            train_predict = scaler.inverse_transform(model.predict(X_train))
            test_predict = scaler.inverse_transform(model.predict(X_test))

            # Reskalowanie rzeczywistych wartości
            train_true = scaler.inverse_transform(y_train)
            test_true = scaler.inverse_transform(y_test)

            # Ocena modelu i rozkład błędów
            st.subheader("Ocena modelu")

            # Obliczanie i wyświetlanie RMSE i MAE dla zbioru treningowego i testowego
            st.write("### Ocena zbioru treningowego")
            if dataset_option == "ETH":
                evaluate_model(train_true, train_predict, dataset_type="Treningowy")
            else:
                evaluate_model_b(train_true, train_predict, dataset_type="Treningowy")

            st.write("### Ocena zbioru testowego")
            if dataset_option == "ETH":
                evaluate_model(test_true, test_predict, dataset_type="Testowy")
            else:
                evaluate_model_b(test_true, test_predict, dataset_type="Testowy")

            # Wykresy rozkładu błędów
            st.write("### Rozkład błędów (dane treningowe)")
            if dataset_option == "ETH":
                plot_error_distribution(train_true, train_predict, dataset_type="Treningowy")
            else:
                plot_error_distribution_b(train_true, train_predict, dataset_type="Treningowy")

            st.write("### Rozkład błędów (dane testowe)")
            if dataset_option == "ETH":
                plot_error_distribution(test_true, test_predict, dataset_type="Testowy")
            else:
                plot_error_distribution_b(test_true, test_predict, dataset_type="Testowy")
    else:
        model_path = f'{dataset_option.lower()}_hybrid.keras'
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.write(f"Załadowano model dla {dataset_option}.")

            time_step = 72
            if dataset_option == "ETH":
                X, y, scaler, df, close_data = load_eth_data(file_path, time_step)
            else:
                X, y, scaler, df, close_data = load_btc_data(file_path, time_step)

            train_size = int(len(y) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Prognozy
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Ocena modelu
            st.write("### Ocena zbioru treningowego")
            if dataset_option == "ETH":
                evaluate_eth_model(model, X_train, y_train, scaler, dataset_name="Treningowy")
            else:
                evaluate_btc_model(model, X_train, y_train, scaler, dataset_name="Treningowy")

            st.write("### Ocena zbioru testowego")
            if dataset_option == "ETH":
                evaluate_eth_model(model, X_test, y_test, scaler, dataset_name="Testowy")
            else:
                evaluate_btc_model(model, X_test, y_test, scaler, dataset_name="Testowy")
        else:
            st.error("Nie znaleziono modelu do oceny. Proszę najpierw wytrenować model.")
else:
    st.warning("Proszę wybrać opcję, aby kontynuować.")
