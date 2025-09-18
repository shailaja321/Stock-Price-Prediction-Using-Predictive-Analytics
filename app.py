import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("📈 Stock Price Prediction App")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker (example: AAPL, MSFT, TSLA)", "AAPL")

# Fetch Data
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
st.write("### Stock Data (last 5 rows)", data.tail())

# Plot Closing Price
st.subheader("Closing Price Chart")
st.line_chart(data['Close'])

# Feature Engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(21).std()
data['Momentum'] = data['Close'].diff()
data = data.dropna()

X = data[['Open', 'High', 'Low', 'Volume', 'Return', 'Volatility', 'Momentum']]
y = data['Close']

# Train Model (use all but last 30 days for training)
model = RandomForestRegressor()
model.fit(X[:-30], y[:-30])
pred = model.predict(X[-30:])

# Fix: Flatten pred to avoid "must be 1D" error
pred = pred.flatten() if pred.ndim > 1 else pred  

# Create Results DataFrame
results_df = pd.DataFrame({
    "Actual": y[-30:].values,
    "Predicted": pred
}, index=y[-30:].index)

# Show Predictions
st.subheader("Predicted vs Actual (Last 30 Days)")
st.line_chart(results_df)

# Extra visualization with Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results_df.index, results_df['Actual'], label="Actual Price", color="blue")
ax.plot(results_df.index, results_df['Predicted'], label="Predicted Price", color="red", linestyle="--")
ax.set_title(f"{ticker} Stock Price Prediction (Last 30 Days)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)
