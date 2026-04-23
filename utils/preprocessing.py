import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker="NVO"):
    data = yf.download(ticker, start="2015-01-01")
    return data

def compute_returns(data):
    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    return data.dropna()

def compute_volatility(data, window=5):
    data["Volatility"] = data["Returns"].rolling(window).std()
    return data.dropna()