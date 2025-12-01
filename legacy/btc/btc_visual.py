import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def visualize_btc(data_path, start_date='2018-01-13'):

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Date', 'Price'])
    df = df.set_index('Date')

    df = df[df.index >= start_date]

    df_hour = df.resample('h').mean()
    df_day = df.resample('D').mean()
    df_month = df.resample('ME').mean()
    df_quarter = df.resample('QE').mean()
    df_year = df.resample('YE').mean()

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle('Kursy wymiany Bitcoin - Średnia USD', fontsize=22, fontweight='bold')

    # Wykres danych godzinnych
    plt.subplot(231)
    plt.plot(df_hour['Price'], '-', label='Godzinowo', color='steelblue', linewidth=1)
    plt.title('Bitcoin na godzinę', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres danych dziennych
    plt.subplot(232)
    plt.plot(df_day['Price'], '-', label='Dziennie', color='dodgerblue', linewidth=2)
    plt.title('Bitcoin na dzień', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres danych miesięcznych
    plt.subplot(233)
    plt.plot(df_month['Price'], '-', label='Miesięcznie', color='tomato', linewidth=2)
    plt.title('Bitcoin na miesiąc', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres danych kwartalnych
    plt.subplot(234)
    plt.plot(df_quarter['Price'], '-', label='Kwartalnie', color='seagreen', linewidth=2)
    plt.title('Bitcoin na kwartał', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Wykres danych rocznych
    plt.subplot(235)
    plt.plot(df_year['Price'], '-', label='Rocznie', color='purple', linewidth=2)
    plt.title('Bitcoin na rok', fontsize=18)
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Cena (USD)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    st.pyplot(fig)
