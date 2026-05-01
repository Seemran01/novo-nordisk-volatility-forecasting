from arch import arch_model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =========================
# GARCH WALK-FORWARD
# =========================
def walk_forward_garch(df, initial_window, step_size, forecast_horizon=22):

    returns = df["Log_Returns"].dropna()

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(returns) - forecast_horizon, step_size):

        train = returns.iloc[:i]
        test = returns.iloc[i:i+forecast_horizon]

        model = arch_model(train, vol='Garch', p=1, q=1, dist='Normal')
        res = model.fit(disp='off')

        forecast = res.forecast(horizon=forecast_horizon)

        pred = forecast.variance.values[-1, :]   
        actual = test.values ** 2                

        preds.extend(pred[:len(test)])
        actuals.extend(actual)
        dates.extend(test.index)

    return np.array(preds), np.array(actuals), dates


# =========================
# HAR WALK-FORWARD
# =========================
def walk_forward_har(df, initial_window, step_size, forecast_horizon=22):

    X = df[['RV_1D', 'RV_5D', 'RV_22D']]
    y = df['Realized_Vol']  

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(df) - forecast_horizon, step_size):

        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i+forecast_horizon]
        y_test = y.iloc[i:i+forecast_horizon]

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        preds.extend(pred)
        actuals.extend(y_test.values)
        dates.extend(y_test.index)

    return np.array(preds), np.array(actuals), dates


# =========================
# NAIVE WALK-FORWARD
# =========================
def walk_forward_naive(series, initial_window, step_size, forecast_horizon=22):

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(series) - forecast_horizon, step_size):

        train = series.iloc[:i]
        test = series.iloc[i:i+forecast_horizon]

        pred = train.iloc[-1]

        preds.extend([pred] * len(test))
        actuals.extend(test.values)
        dates.extend(test.index)

    return np.array(preds), np.array(actuals), dates


# =========================
# EWMA WALK-FORWARD
# =========================
def walk_forward_ewma(returns, initial_window, step_size, forecast_horizon=22, lam=0.94):

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(returns) - forecast_horizon, step_size):

        train = returns.iloc[:i]
        test = returns.iloc[i:i+forecast_horizon]

        var = train.var()

        for r in train:
            var = lam * var + (1 - lam) * (r ** 2)

        pred = np.array([var] * len(test))

        preds.extend(pred)
        actuals.extend(test.values ** 2)   
        dates.extend(test.index)

    return np.array(preds), np.array(actuals), dates



def forecast_garch_next(df):

    returns = df["Log_Returns"].dropna()

    model = arch_model(returns, vol="Garch", p=1, q=1, dist="Normal")
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1)

    return forecast.variance.values[-1, 0]


def forecast_har_next(model, df):

    X = df[['RV_1D', 'RV_5D', 'RV_22D']].iloc[-1].values.reshape(1, -1)

    return model.predict(X)[0]


def forecast_ewma_next(returns, lam=0.94):

    var = returns.var()

    for r in returns:
        var = lam * var + (1 - lam) * (r ** 2)

    return float(var)    



def forecast_naive_next(df):

    return df["Realized_Vol"].iloc[-1]