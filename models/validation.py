import numpy as np
import pandas as pd

def walk_forward_validation(
    X,
    y,
    model_fn,
    initial_window=756,
    step_size=5,
    forecast_horizon=22
):

    preds = []
    actuals = []
    dates = []

    start = initial_window
    n = len(X)

    for i in range(start, n - forecast_horizon, step_size):

        # ------------------------
        # EXPANDING TRAIN SET
        # ------------------------
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i + forecast_horizon]
        y_test = y.iloc[i:i + forecast_horizon]

        # ------------------------
        # TRAIN MODEL
        # ------------------------
        model = model_fn()
        model.fit(X_train, y_train)

        # ------------------------
        # PREDICT
        # ------------------------
        y_pred = model.predict(X_test)

        preds.extend(y_pred)
        actuals.extend(y_test.values)
        dates.extend(y_test.index)

    return np.array(preds), np.array(actuals), dates



def forecast_ml_next_day(model, df, feature_cols):

    last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)

    return model.predict(last_row)[0]