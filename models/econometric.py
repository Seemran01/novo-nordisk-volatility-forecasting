# models/econometric.py
from arch import arch_model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_garch(df, test_size=0.2):
    returns = df['Log_Returns'].dropna()

    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')

    dates = returns.index

    vol = pd.Series(res.conditional_volatility, index=dates)

    split = int(len(vol) * (1 - test_size))

    vol_test = vol.iloc[split:]
    test_dates = vol_test.index

    return vol_test.values, test_dates


# In run_har, return the index too:
def run_har(df, test_size=0.2):
    har_features = ['RV_1D', 'RV_5D', 'RV_22D']

    X = df[har_features].values
    y = df['Realized_Vol'].values
    dates = df.index  # ✅ capture dates

    valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid_idx], y[valid_idx]
    dates = dates[valid_idx]  # ✅ filter dates too

    if len(X) < 30:
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = dates[split:]  # ✅ test period dates

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model.predict(X_test), y_test, test_dates  # ✅ return dates

def naive_persistence(rv_series):
    return rv_series.shift(1)

def ewma_volatility(returns, lam=0.94):
    ewma_var = []
    var = returns.var()

    for r in returns:
        var = lam * var + (1 - lam) * (r ** 2)
        ewma_var.append(var)

    ewma_series = pd.Series(ewma_var, index=returns.index)

    # ✅ return EXACTLY 3 values
    return ewma_series.values, returns.values, returns.index