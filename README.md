# Prognozowanie Kursów Kryptowalut za pomocą AI

## Opis projektu

Projekt wykorzystuje metody sztucznej inteligencji (AI) do predykcji kursów kryptowalut, takich jak Ether (ETH) i Bitcoin (BTC). W ramach narzędzia zaimplementowano dwa modele:
- **LSTM (Long Short-Term Memory)**: Model sekwencyjny skuteczny w analizie danych szeregów czasowych.
- **CNN-LSTM (Hybrid)**: Połączenie konwolucyjnych sieci neuronowych (CNN) i LSTM dla bardziej zaawansowanej analizy danych.

Aplikacja pozwala na wizualizację danych, trenowanie, bądź wykorzystanie wcześniej wytrenowanych modeli, ocenę ich wydajności oraz generowanie prognoz przyszłych cen kryptowalut.

## Funkcjonalności
1. **Wizualizacja danych historycznych**:
   - Przegląd danych cenowych ETH i BTC.
   - Wizualizacja trendów i anomalii w danych.
2. **Trenowanie modeli AI**:
   - Możliwość wyboru między modelem LSTM i CNN-LSTM.
   - Zapis wytrenowanych modeli w celu późniejszego użycia.
3. **Prognozowanie przyszłych cen**:
   - Generowanie prognoz cen kryptowalut na określoną liczbę dni.
   - Wizualizacja prognoz na wykresach.
4. **Ewaluacja modeli**:
   - Obliczanie metryk, takich jak RMSE, MAE, R^2.
   - Wizualizacja histogramów rozkładu błędów modeli.

## Struktura projektu

- **Pliki z modelami:**
  - `ETH_lstm.py` – implementacja modelu LSTM dla kryptowaluty Ether.
  - `BTC_lstm.py` – implementacja modelu LSTM dla Bitcoina.
  - `ETH_hybrid.py` – implementacja modelu CNN-LSTM dla kryptowaluty Ether.
  - `BTC_hybrid.py` – implementacja modelu CNN-LSTM dla Bitcoina.
- **Wizualizacja danych:**
  - `eth_visual.py` – wizualizacja danych Ether.
  - `btc_visual.py` – wizualizacja danych Bitcoina.
- **Dane wejściowe:**
  - `eth_data_cleaned.csv` – dane historyczne cen Ether.
  - `btc_data.csv` – dane historyczne cen Bitcoina.
- **Główna aplikacja:**
  - `interface.py` – interfejs aplikacji zbudowany w Streamlit.

## Wymagania systemowe

- Python 3.8 lub nowszy.
- Zainstalowane pakiety Python:
  - `streamlit`
  - `keras`
  - `tensorflow`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `os`
  - `datetime`
-

## Instalacja

1. Sklonuj projekt lub pobierz pliki.
2. Zainstaluj wymagane pakiety.
3. Uruchom za pomocą komendy: streamlit run interface.py

