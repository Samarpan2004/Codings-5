# stock_lstm.py
# Run: python stock_lstm.py AAPL 60
# downloads history, trains small LSTM to predict next-day close (toy example)

import sys, yfinance as yf
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def create_dataset(series, window=60):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def main():
    ticker = sys.argv[1] if len(sys.argv)>1 else "AAPL"
    window = int(sys.argv[2]) if len(sys.argv)>2 else 60
    data = yf.download(ticker, period="3y", interval="1d")
    close = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler(); close_s = scaler.fit_transform(close)
    X, y = create_dataset(close_s.flatten(), window)
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential([LSTM(50, return_sequences=True, input_shape=(window,1)), LSTM(50), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    preds = model.predict(X_test)
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.legend(); plt.show()
    model.save(f"models/{ticker}_lstm.h5")
    print("Saved model")

if __name__=="__main__":
    main()
