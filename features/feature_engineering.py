import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy()
    n = len(df)

    # -----------------------
    # RETURNS
    # -----------------------
    df['Log_Returns'] = np.log(df['Close']).diff()

    # -----------------------
    # TARGET (NEXT-DAY PROXY)
    # -----------------------
    df['Realized_Vol'] = df['Log_Returns']**2
    df['Realized_Vol'] = df['Realized_Vol'].clip(lower=1e-8)

    # -----------------------
    # ML FEATURES
    # -----------------------
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Volatility_10'] = df['Log_Returns'].rolling(10).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    # -----------------------
    # HAR FEATURES (STRICTLY LAGGED)
    # -----------------------
    r2 = df['Log_Returns']**2

    df['RV_1D'] = r2.shift(1)
    df['RV_5D'] = r2.rolling(5).mean().shift(1)
    df['RV_22D'] = r2.rolling(22).mean().shift(1)

    # -----------------------
    # CLEANING
    # -----------------------
    df.dropna(inplace=True)

    return df