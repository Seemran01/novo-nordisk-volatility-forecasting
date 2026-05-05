import pandas as pd

def prepare_data(df):

    feature_cols = [
        'Log_Returns',
        'MA_5',
        'MA_20',
        'Volatility_10',
        'Volume_Change',
        'RV_1D',
        'RV_5D',
        'RV_22D'
    ]

    X = df[feature_cols].ffill().bfill()
    y = df["Realized_Vol"]

    return X, y