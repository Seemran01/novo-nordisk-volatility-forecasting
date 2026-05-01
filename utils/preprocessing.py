import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------
# 1. DOWNLOAD DATA
# ---------------------------
def download_data(ticker="NVO", start="2015-01-01"):
    data = yf.download(ticker, start=start)
    return data

# ---------------------------
# 2. RETURNS
# ---------------------------
def compute_returns(data):
    data = data.copy()
    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    return data

# ---------------------------
# 3. VOLATILITY
# ---------------------------
def compute_volatility(data, window=5):
    data = data.copy()

    data["Volatility"] = data["Returns"].rolling(
        window=window,
        min_periods=window
    ).std()

    return data

# ---------------------------
# 4. FINAL CLEANING
# ---------------------------
def clean_data(data):
    # only drop rows where core variables are missing
    data = data.dropna(subset=["Close", "Returns", "Volatility"])
    return data

