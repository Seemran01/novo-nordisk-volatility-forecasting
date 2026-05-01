def prepare_data(df):
    feature_cols = [
        'Open','High','Low','Close','Volume',
        'MA_5','MA_20','Volatility_10','Volume_Change'
    ]

    X = df[feature_cols].ffill().bfill()

    y = df["Realized_Vol"]

    return X, y