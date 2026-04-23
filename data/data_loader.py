import yfinance as yf
import streamlit as st

@st.cache_data
def load_data(ticker="NVO", period="1y"):
    df = yf.download(ticker, period=period)

    if df.empty:
        return None

    return df