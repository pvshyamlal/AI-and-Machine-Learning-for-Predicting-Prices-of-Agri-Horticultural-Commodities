import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st

file_path = "DatasetSIH1647.csv" 
df = pd.read_csv(file_path)

df.set_index('Commodities', inplace=True)
df = df.T
df.index = pd.date_range(start='2014', periods=len(df), freq='YE')

df = df.ffill()

commodities = df.columns.tolist()

st.title("Commodity Price Forecasting")

selected_commodity = st.selectbox("Choose a Commodity", commodities)

if st.button("Submit"):
    data = df[selected_commodity]

    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    sarimax_model = model.fit(disp=False)
    # Forecast for the next 5 years
    forecast = sarimax_model.get_forecast(steps=5)
    forecasted_values = forecast.predicted_mean

    forecast_years = pd.date_range(start='2025', periods=5, freq='YE')
    forecast_df = pd.DataFrame({'Year': forecast_years, f'{selected_commodity}_Price_Forecast': forecasted_values})

    st.write(f"### {selected_commodity} Price Forecast (2025-2029)")
    st.write(forecast_df)

    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'Actual {selected_commodity} Prices')
    plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
    plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {train_rmse:.4f}")