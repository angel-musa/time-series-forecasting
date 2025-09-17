import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import streamlit as st

def forecast_arima(ticker):
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Add error handling for yfinance download
    try:
        # Download with longer timeout and retry
        raw_data = yf.download(ticker, start='2021-01-01', end=end_date, timeout=30, progress=False)
        
        # Check if data is empty
        if raw_data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the ticker symbol.")
            return None, None, None, None
        
        # Handle different DataFrame structures from yfinance
        # Check if columns are MultiIndex (happens with single ticker sometimes)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take the first level that contains 'Adj Close'
            raw_data.columns = raw_data.columns.get_level_values(0)
        
        # Now try to get Adj Close data with fallback options
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            # Fallback to Close if Adj Close not available
            data = raw_data['Close']
            st.info(f"Using Close price instead of Adjusted Close for {ticker}")
        else:
            # Debug: show what columns are available
            st.error(f"Neither 'Adj Close' nor 'Close' found for {ticker}. Available columns: {list(raw_data.columns)}")
            return None, None, None, None
        
        # Check if we have enough data points
        if len(data) < 100:
            st.error(f"Insufficient data for ticker {ticker}. Need at least 100 data points.")
            return None, None, None, None
            
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {str(e)}")
        return None, None, None, None

    # Ensure the date index has a frequency
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('B')

    # Forward fill any missing values
    data = data.ffill()

    # Split the data: 66% training, 34% testing
    train_size = int(len(data) * 0.66)
    train, test = data[:train_size], data[train_size:]

    # Define ARIMA order based on auto_arima results
    p, d, q = 1, 1, 0

    try:
        # Fit the SARIMAX model with manually specified (p, d, q)
        manual_model = SARIMAX(train, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
        manual_model_fit = manual_model.fit(disp=False)

        # Make predictions on the test set
        start = len(train)
        end = len(train) + len(test) - 1
        manual_predictions = manual_model_fit.predict(start=start, end=end, dynamic=False)
        manual_predictions = pd.Series(manual_predictions, index=test.index)

        # Drop any NaN values from test and predictions
        test = test.dropna()
        manual_predictions = manual_predictions.dropna()

        # Calculate error
        manual_error = mean_squared_error(test, manual_predictions)
        st.write(f'Test MSE (manual ARIMA): {manual_error:.3f}')

        # Retrain the model on the entire dataset (train + test)
        manual_model_fit = SARIMAX(data, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

        # Forecast the next 7 days from the end of the data
        manual_forecast = manual_model_fit.forecast(steps=7)
        manual_forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7, freq='B')
        manual_forecast = pd.Series(manual_forecast, index=manual_forecast_dates)

        # Calculate residuals
        manual_residuals = test - manual_predictions

        # Plot residuals and density plot
        residuals_fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        manual_residuals.plot(title='Residuals', ax=axes[0])
        manual_residuals.plot(kind='kde', title='Density', ax=axes[1])  # Kernel density estimation for density plot
        residuals_fig.tight_layout()  # Ensure layout is neat

        # ACF and PACF plots
        acf_fig, acf_ax = plt.subplots(figsize=(10, 6))
        plot_acf(manual_residuals, ax=acf_ax)
        acf_ax.set_title('Autocorrelation Function')

        pacf_fig, pacf_ax = plt.subplots(figsize=(10, 6))
        plot_pacf(manual_residuals, ax=pacf_ax)
        pacf_ax.set_title('Partial Autocorrelation Function')

        # Plot the results
        final_forecast_fig, final_ax = plt.subplots(figsize=(12, 6))
        final_ax.plot(data, label='Historical Prices')
        final_ax.plot(test.index, manual_predictions, label='Predicted Prices', color='orange')
        final_ax.plot(manual_forecast_dates, manual_forecast, label='Forecast Prices', linestyle='--', marker='o', color='red')
        final_ax.set_title(f'{ticker} Stock Price Prediction')
        final_ax.set_xlabel('Date')
        final_ax.set_ylabel('Price')
        final_ax.legend()

        # Return the figures for display in the main app
        return final_forecast_fig, acf_fig, pacf_fig, residuals_fig
        
    except Exception as e:
        st.error(f"Error in ARIMA modeling: {str(e)}")
        return None, None, None, None