from sklearn.preprocessing import StandardScaler
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

    X_raw = df[feature_cols].ffill().bfill()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Converts back to DataFrame
    X = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    y = df["Realized_Vol"]

    return X, y