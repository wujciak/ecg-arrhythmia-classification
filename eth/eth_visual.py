import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def visualize_eth(data_path, start_date='2017-01-01'):

    df_eth = pd.read_csv(data_path)
    df_eth['Date'] = pd.to_datetime(df_eth['Date'], format='%m/%d/%Y', errors='coerce')
    df_eth = df_eth.dropna(subset=['Date', 'Price'])
    df_eth = df_eth.set_index('Date')

    numeric_columns = ['Price', 'Open', 'High', 'Low']
    for col in numeric_columns:
        df_eth[col] = pd.to_numeric(df_eth[col], errors='coerce')

    def convert_volume(vol):
        if isinstance(vol, str):
            vol = vol.replace('K', 'e3').replace('M', 'e6')
        return pd.to_numeric(vol, errors='coerce')

    df_eth['Volume'] = df_eth['Volume'].apply(convert_volume)

    if 'Change %' in df_eth.columns:
        df_eth['Change %'] = df_eth['Change %'].astype(str)
        df_eth['Change %'] = df_eth['Change %'].str.replace('%', '')
        df_eth['Change %'] = pd.to_numeric(df_eth['Change %'], errors='coerce')

    df_eth = df_eth.resample('D').mean()
    df_eth = df_eth[df_eth.index >= start_date]
    df_eth_month = df_eth.resample('ME').mean()
    df_eth_year = df_eth.resample('YE').mean()
    df_eth_Q = df_eth.resample('QE').mean()

    # Rysowanie wykresów
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle('Kursy wymiany Ether - Średnia USD', fontsize=22, fontweight='bold')

    # Wykres na podstawie danych dziennych
    plt.subplot(221)
    plt.plot(df_eth['Price'], '-', label='Dziennie', color='dodgerblue', linewidth=2)
    plt.title('Ether na dzień', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres na podstawie danych miesięcznych
    plt.subplot(222)
    plt.plot(df_eth_month['Price'], '-', label='Miesięcznie', color='tomato', linewidth=2)
    plt.title('Ether na miesiąc', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres danych kwartalnych
    plt.subplot(223)
    plt.plot(df_eth_Q['Price'], '-', label='Kwartalnie', color='seagreen', linewidth=2)
    plt.title('Ether na kwartał', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres na podstawie danych rocznych
    plt.subplot(224)
    plt.plot(df_eth_year['Price'], '-', label='Rocznie', color='purple', linewidth=2)
    plt.title('Ether na rok', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    st.pyplot(fig)
    plt.show()
