import streamlit as st
import yfinance as yf
import importlib
import matplotlib.pyplot as plt
from datetime import datetime
from arima import forecast_arima
from boost import forecast_boost
from prophet_model import forecast_prophet
from lstm import forecast_lstm

# Function to display explanations for ACF and PACF plots
def display_plots_and_explanations(acf_fig, pacf_fig, residuals_fig):
    if acf_fig is not None and pacf_fig is not None and residuals_fig is not None:
        # Display ACF and PACF plots
        st.pyplot(acf_fig)
        st.write("""
        **Autocorrelation Function (ACF)**: The ACF plot helps identify the correlation between a time series and its lagged values. Significant spikes in the ACF indicate the presence of autocorrelation, which suggests that past values have an influence on current values.
        """)

        st.pyplot(pacf_fig)
        st.write("""
        **Partial Autocorrelation Function (PACF)**: The PACF plot is used to determine the extent of lag that influences the time series. The first significant lag indicates how many past values should be included in the model. 
        """)

        # Display residuals plot
        st.pyplot(residuals_fig)
        st.write("""
        **Residuals Plot**: The residuals plot shows the difference between the predicted and actual values. Ideally, the residuals should be randomly scattered around zero, indicating that the model has captured all underlying patterns in the data. Patterns in the residuals may suggest that the model can be improved. 
        """)

# Main screen for ticker search
st.title("Time-Series Forecasting Dashboard")

# Model selection section
st.subheader("Model Selection")
model_option = st.selectbox("Choose forecasting model:", ("ARIMA", "XGBoost", "Prophet", "LSTM"))

# Ticker input section
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
end_date = datetime.now().strftime('%Y-%m-%d')

if ticker:
    # Fetching data from yfinance with error handling
    try:
        data = yf.download(ticker, start="2021-01-01", end=end_date, progress=False)
        
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the ticker symbol.")
        else:
            st.write(f"Stock data for {ticker}")
            st.line_chart(data['Close'])

            # Short-term forecasting
            st.subheader("Short-Term Forecast")

            # Load the selected model's code from the corresponding file
            if model_option == "ARIMA":
                with st.spinner(f'Training ARIMA model for {ticker}... This may take a few moments.'):
                    results = forecast_arima(ticker)
                if results and all(result is not None for result in results):
                    final_forecast_fig, acf_fig1, pacf_fig1, residuals_fig1 = results
                    st.pyplot(final_forecast_fig)
                    display_plots_and_explanations(acf_fig1, pacf_fig1, residuals_fig1)
                    
            elif model_option == "XGBoost":
                with st.spinner(f'Training XGBoost model for {ticker}... This may take a few moments.'):
                    results = forecast_boost(ticker)
                if results and all(result is not None for result in results):
                    final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = results
                    st.pyplot(final_forecast_fig)
                    display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
                    
            elif model_option == "Prophet":
                with st.spinner(f'Training Prophet model for {ticker}... This may take a few moments.'):
                    results = forecast_prophet(ticker)
                if results and all(result is not None for result in results):
                    final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = results
                    st.plotly_chart(final_forecast_fig)
                    display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
                    
            elif model_option == "LSTM":
                with st.spinner(f'Training LSTM model for {ticker}... This may take a few moments. Training neural network...'):
                    results = forecast_lstm(ticker)
                if results and all(result is not None for result in results):
                    final_forecast_fig, acf_fig2, pacf_fig2, residuals_fig2 = results
                    st.plotly_chart(final_forecast_fig)
                    display_plots_and_explanations(acf_fig2, pacf_fig2, residuals_fig2)
                    
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {str(e)}")
        st.info("Please try again or check your internet connection.")