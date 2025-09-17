import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
import optuna
from sklearn.ensemble import BaggingRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st

def forecast_boost(stock_symbol):
    try:
        # Load data with error handling
        start_date = '2021-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        raw_data = yf.download(stock_symbol, start=start_date, end=end_date, timeout=30, progress=False)
        
        # Check if data is empty
        if raw_data.empty:
            st.error(f"No data found for ticker {stock_symbol}. Please check the ticker symbol.")
            return None, None, None, None
        
        # Handle different DataFrame structures from yfinance
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        
        # Get Close price data
        if 'Close' in raw_data.columns:
            close_data = raw_data['Close']
        elif 'Adj Close' in raw_data.columns:
            close_data = raw_data['Adj Close']
        else:
            st.error(f"No Close or Adj Close data found for {stock_symbol}. Available columns: {list(raw_data.columns)}")
            return None, None, None, None
        
        # Reset index and prepare data
        data = pd.DataFrame({
            'ds': raw_data.index,
            'y': close_data.values
        }).reset_index(drop=True)
        
        data['unique_id'] = stock_symbol  # Add unique_id column

        # Check if we have enough data
        if len(data) < 100:
            st.error(f"Insufficient data for ticker {stock_symbol}. Need at least 100 data points.")
            return None, None, None, None

        # Train-test split
        train_size = int(len(data) * 0.66)
        train_data = data[:train_size].copy()
        test_data = data[train_size:].copy()

        # Manual Feature Engineering
        def create_lags(df, lags):
            df = df.copy()
            for lag in lags:
                df[f'lag_{lag}'] = df['y'].shift(lag)
            return df

        def create_ewm(df, spans):
            df = df.copy()
            for span in spans:
                df[f'ewm_{span}'] = df['y'].ewm(span=span).mean()
            return df

        def add_date_parts(df):
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            df['year'] = df['ds'].dt.year
            df['month'] = df['ds'].dt.month
            df['day'] = df['ds'].dt.day
            df['weekday'] = df['ds'].dt.weekday
            return df

        # Feature engineering for train and test data
        train_data = create_lags(train_data, lags=[1, 2, 3, 4, 5, 6, 7])
        train_data = create_ewm(train_data, spans=[7, 14, 21])
        train_data = add_date_parts(train_data)
        train_data = train_data.dropna().reset_index(drop=True)  # Drop rows with NaN values

        test_data = create_lags(test_data, lags=[1, 2, 3, 4, 5, 6, 7])
        test_data = create_ewm(test_data, spans=[7, 14, 21])
        test_data = add_date_parts(test_data)
        test_data = test_data.dropna().reset_index(drop=True)  # Drop rows with NaN values

        # Define feature columns - fix the tuple issue
        feature_columns = []
        for col in train_data.columns:
            if isinstance(col, str):  # Ensure column name is string
                if col.startswith('lag_') or col.startswith('ewm_') or col in ['year', 'month', 'day', 'weekday']:
                    feature_columns.append(col)

        if not feature_columns:
            st.error("No valid feature columns found for modeling.")
            return None, None, None, None

        # Define the objective function for Bayesian Optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True)
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Train the model
            model.fit(train_data[feature_columns], train_data['y'])
            
            # Predict
            predictions = model.predict(test_data[feature_columns])
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test_data['y'], predictions))
            return rmse

        # Optimize hyperparameters with fewer trials to speed up
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        # Initialize the XGBoost model with best parameters
        best_params = study.best_params
        xgb_model = xgb.XGBRegressor(**best_params)
        model = BaggingRegressor(xgb_model, n_estimators=30)

        # Fit the model with the best parameters
        model.fit(train_data[feature_columns], train_data['y'])

        # Predict on the test_data range
        predictions = model.predict(test_data[feature_columns])

        # Calculate residuals
        residuals = test_data['y'] - predictions

        # Create ACF and PACF plots
        acf_fig, acf_ax = plt.subplots(figsize=(10, 6))
        plot_acf(residuals, ax=acf_ax)
        acf_ax.set_title('Autocorrelation Function of Residuals')

        pacf_fig, pacf_ax = plt.subplots(figsize=(10, 6))
        plot_pacf(residuals, ax=pacf_ax)
        pacf_ax.set_title('Partial Autocorrelation Function of Residuals')

        # Create residuals plot
        residuals_fig, residuals_ax = plt.subplots(figsize=(10, 6))
        residuals_ax.plot(residuals)
        residuals_ax.axhline(0, color='red', linestyle='--')
        residuals_ax.set_title('Residuals Plot')
        residuals_ax.set_ylabel('Residuals')

        # Define forecast horizon (number of days you want to forecast beyond test_data)
        forecast_horizon = 30  # Example: forecast 30 days into the future

        # Create future dates
        future_dates = pd.date_range(start=data['ds'].iloc[-1] + timedelta(days=1), periods=forecast_horizon)
        future_data = pd.DataFrame({
            'ds': future_dates,
            'unique_id': stock_symbol,
            'y': data['y'].iloc[-1]  # Initialize with last known value
        })

        # Add feature engineering to future data
        # For simplicity, use last known values for lag features
        for lag in [1, 2, 3, 4, 5, 6, 7]:
            future_data[f'lag_{lag}'] = data['y'].iloc[-lag] if len(data) >= lag else data['y'].iloc[-1]
        
        for span in [7, 14, 21]:
            future_data[f'ewm_{span}'] = data['y'].iloc[-span:].mean() if len(data) >= span else data['y'].iloc[-1]
        
        future_data = add_date_parts(future_data)

        # Ensure all feature columns exist in future_data
        for col in feature_columns:
            if col not in future_data.columns:
                future_data[col] = 0

        # Predict future prices
        future_predictions = model.predict(future_data[feature_columns])

        # Plot the results
        forecast_fig, forecast_ax = plt.subplots(figsize=(14, 7))
        
        # Plot actual prices
        forecast_ax.plot(data['ds'], data['y'], label='Actual Prices', color='blue')

        # Plot test data (actual prices within the test period)
        forecast_ax.plot(test_data['ds'], test_data['y'], label='Test Data', color='orange')

        # Plot model predictions
        forecast_ax.plot(test_data['ds'], predictions, label='Predicted Prices', color='red')

        # Plot future forecast prices
        forecast_ax.plot(future_dates, future_predictions, label='Future Forecast', color='green', linestyle='--')

        forecast_ax.set_title(f'{stock_symbol} Stock Price Forecast')
        forecast_ax.set_xlabel('Date')
        forecast_ax.set_ylabel('Price')
        forecast_ax.legend()
        forecast_ax.grid(True)

        return forecast_fig, acf_fig, pacf_fig, residuals_fig
        
    except Exception as e:
        st.error(f"Error in XGBoost modeling: {str(e)}")
        return None, None, None, None