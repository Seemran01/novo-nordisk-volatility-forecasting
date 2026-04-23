import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy()
    n = len(df)

    df['Log_Returns']     = np.log(df['Close'] / df['Close'].shift(1))
    df['Squared_Returns'] = df['Log_Returns'] ** 2

    # ✅ Realized Vol: simple rolling std — consistent, interpretable scale
    df['Realized_Vol']    = df['Log_Returns'].rolling(min(22, n)).std()

    # ML features
    df['MA_5']            = df['Close'].rolling(min(5, n)).mean()
    df['MA_20']           = df['Close'].rolling(min(20, n)).mean()
    df['Volatility_10']   = df['Log_Returns'].rolling(min(10, n)).std()
    df['Volume_Change']   = df['Volume'].pct_change()

    # HAR features — lagged Realized_Vol, same scale
    df['RV_1D']  = df['Realized_Vol'].shift(1)
    df['RV_5D']  = df['Realized_Vol'].rolling(min(5, n)).mean().shift(1)
    df['RV_22D'] = df['Realized_Vol'].rolling(min(22, n)).mean().shift(1)

    df.dropna(inplace=True)
    return df