import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("📈 Stock Price Prediction App")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch Data
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
st.write("### Stock Data", data.tail())

# Plot Closing Price
st.line_chart(data['Close'])

# Feature Engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(21).std()
data['Momentum'] = data['Close'].diff()
data = data.dropna()

X = data[['Open', 'High', 'Low', 'Volume', 'Return', 'Volatility', 'Momentum']]
y = data['Close']

# Train Model
model = RandomForestRegressor()
model.fit(X[:-30], y[:-30])
pred = model.predict(X[-30:])

# Show Predictions
st.subheader("Predicted vs Actual")

# Ensure predictions are 1D
pred = pred.flatten() if pred.ndim > 1 else pred  

# Now create DataFrame for chart
results_df = pd.DataFrame({
    "Actual": y[-30:].values, 
    "Predicted": pred
}, index=y[-30:].index)

st.subheader("Predicted vs Actual")
st.line_chart(results_df)
