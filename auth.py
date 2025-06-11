# -------------------- Required Libraries -------------------- #
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# -------------------- Set Page Configuration -------------------- #
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# -------------------- Stock Overview -------------------- #
st.title("📈 Stock Market Prediction")

stock_ticker = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(stock_ticker, start="2010-01-01", end="2025-12-31")

if not df.empty and "Close" in df.columns:
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Display dataset
    st.subheader("Stock Data")
    st.write(df)

    # Closing Price Chart
    st.subheader("Closing Price vs Time")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Close'])
    st.pyplot(fig)

    # Moving Averages (100 & 200 MA)
    st.subheader("Closing Price with 100 MA & 200 MA")
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig_ma, ax_ma = plt.subplots(figsize=(12,6))
    ax_ma.plot(ma100, label="100 MA", color='red')
    ax_ma.plot(ma200, label="200 MA", color='green')
    ax_ma.plot(df['Close'], label="Closing Price", color='blue')
    ax_ma.legend()
    st.pyplot(fig_ma)

    # RSI Calculation
    st.subheader("Relative Strength Index (RSI)")
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    fig_rsi, ax_rsi = plt.subplots(figsize=(12,6))
    ax_rsi.plot(rsi, label="RSI", color='purple')
    ax_rsi.axhline(30, linestyle='--', color='red')  # Oversold
    ax_rsi.axhline(70, linestyle='--', color='green')  # Overbought
    ax_rsi.legend()
    st.pyplot(fig_rsi)

    # -------------------- AI-Powered Predictions -------------------- #
    st.subheader("Stock Price Predictions (LSTM & ARIMA)")

    # LSTM Prediction
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)
    model = load_model('keras_model.keras')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted *= scale_factor
    y_test = np.array(y_test) * scale_factor

    fig_lstm = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label="LSTM Predicted Price")
    plt.legend()
    st.pyplot(fig_lstm)

    # ARIMA Forecasting
    st.subheader("ARIMA Forecast")
    model_arima = ARIMA(df['Close'], order=(5,1,0))
    model_arima_fit = model_arima.fit()
    forecast_steps = 30
    forecast = model_arima_fit.forecast(steps=forecast_steps)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

    fig_arima = plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label="Actual Prices", color='blue')
    plt.plot(future_dates, forecast, label="ARIMA Forecast", color='purple')
    plt.legend()
    st.pyplot(fig_arima)

else:
    st.warning("Invalid stock ticker or no data available. Please try another ticker.")