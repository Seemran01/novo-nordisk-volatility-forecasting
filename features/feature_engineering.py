import numpy as np
import pandas as pd

def create_features(df):
    df = df.copy()
    n = len(df)

    # -----------------------
    # CORE VARIABLES
    # -----------------------
    df['Log_Returns'] = np.log(df['Close']).diff()

    df['Realized_Vol'] = df['Log_Returns'].rolling(22).var()
    df['Realized_Vol'] = df['Realized_Vol'].clip(lower=1e-8)

    # -----------------------
    # ML FEATURES
    # -----------------------
    df['MA_5'] = df['Close'].rolling(min(5, n)).mean()
    df['MA_20'] = df['Close'].rolling(min(20, n)).mean()
    df['Volatility_10'] = df['Log_Returns'].rolling(min(10, n)).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    # -----------------------
    # HAR FEATURES
    # -----------------------
    df['RV_1D'] = df['Realized_Vol'].shift(1)
    df['RV_5D'] = df['Realized_Vol'].rolling(min(5, n)).mean().shift(1)
    df['RV_22D'] = df['Realized_Vol'].rolling(min(22, n)).mean().shift(1)

    df.dropna(inplace=True)

    return df