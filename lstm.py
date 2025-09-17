import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from datetime import date, timedelta
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout
from keras._tf_keras.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas_ta as ta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys
import streamlit as st

def forecast_lstm(ticker):
    try:
        # Download stock data with error handling
        end_date = datetime.now().strftime('%Y-%m-%d')
        raw_data = yf.download(ticker, start="2021-01-01", end=end_date, timeout=30, progress=False)

        # Check if data is empty
        if raw_data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the ticker symbol.")
            return None, None, None, None

        # Handle different DataFrame structures from yfinance
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        # Get Adj Close data with fallback
        if 'Adj Close' in raw_data.columns:
            azn_df = raw_data[['Adj Close']].copy()
        elif 'Close' in raw_data.columns:
            azn_df = raw_data[['Close']].copy()
            azn_df.columns = ['Adj Close']  # Rename for consistency
            st.info(f"Using Close price instead of Adjusted Close for {ticker}")
        else:
            st.error(f"No Close or Adj Close data found for {ticker}. Available columns: {list(raw_data.columns)}")
            return None, None, None, None

        # Check if we have enough data
        if len(azn_df) < 100:
            st.error(f"Insufficient data for ticker {ticker}. Need at least 100 data points.")
            return None, None, None, None

        # Prepare data
        azn_adj = azn_df[['Adj Close']]
        azn_adj_arr = azn_adj.values
        training_data_len = int(0.8 * len(azn_adj_arr))

        # Create train and test data sets
        train = azn_adj_arr[:training_data_len]
        test = azn_adj_arr[training_data_len:]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train)

        # Create training data structure with 60 time-steps
        X_train, y_train = [], []
        for i in range(60, len(train_scaled)):
            X_train.append(train_scaled[i-60:i, 0])
            y_train.append(train_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)

        # Prepare test data
        total_data = np.concatenate((train, test), axis=0)
        inputs = total_data[len(total_data) - len(test) - 60:]
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict and inverse transform the predictions
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)

        # Evaluate the model
        rmse = np.sqrt(np.mean((predictions - test)**2))
        st.write(f"Root Mean Squared Error: {rmse:.2f}")

        # Prepare data for plotting
        train_data = azn_adj[:training_data_len]
        test_data = azn_adj[training_data_len:].copy()
        test_data['Predictions'] = predictions

        # Create the Plotly figure
        fig = go.Figure()

        # Add traces for the training data
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Adj Close'], mode='lines', name='Training'))

        # Add traces for the actual test data
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Adj Close'], mode='lines', name='Actual'))

        # Add traces for the predicted test data
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Predictions'], mode='lines', name='Predicted'))

        # Update the layout of the figure
        fig.update_layout(
            title=ticker + " Time Series Analysis",
            xaxis_title="Year",
            yaxis_title="Stock Price",
            legend_title="Legend",
            template="plotly_white"
        )

        fig.update_layout(
            hovermode='x unified',
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
            )
        )

        fig.update_traces(
            hoverinfo="x+y",
            mode='lines',
        )

        # Predict Adjusted Close price for the next 5 days
        pred_prices = []
        last_60_days = azn_adj[-60:].values

        # Calculate ATR if we have OHLC data
        atr = None
        if all(col in raw_data.columns for col in ['High', 'Low', 'Close']):
            atr = ta.atr(raw_data['High'], raw_data['Low'], raw_data['Close'], length=14).iloc[-1]

        for _ in range(5):
            # Scale the last 60 days data
            last_60_days_scaled = scaler.transform(last_60_days)

            # Prepare the input to the model
            X_test_future = np.array([last_60_days_scaled])
            X_test_future = np.reshape(X_test_future, (X_test_future.shape[0], X_test_future.shape[1], 1))

            # Predict the next day's price
            pred_price = model.predict(X_test_future, verbose=0)
            pred_price = scaler.inverse_transform(pred_price)

            # Append the prediction to the list
            pred_prices.append(pred_price[0][0])

            # Update the last 60 days data by removing the oldest value and adding the predicted price
            last_60_days = np.append(last_60_days[1:], pred_price, axis=0)

        # Residuals
        residuals = test_data['Adj Close'].values - test_data['Predictions'].values.flatten()

        # Plot ACF and PACF of residuals
        fig_acf = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_acf(residuals, lags=20)
        plt.title('ACF of Residuals')

        fig_pacf = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_pacf(residuals, lags=20)
        plt.title('PACF of Residuals')

        fig_residuals = plt.figure(figsize=(10, 5))
        plt.plot(residuals)
        plt.title('Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.axhline(0, color='red', linestyle='--')

        return fig, fig_acf, fig_pacf, fig_residuals

    except Exception as e:
        st.error(f"Error in LSTM modeling: {str(e)}")
        return None, None, None, None