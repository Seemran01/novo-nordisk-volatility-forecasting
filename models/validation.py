from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def walk_forward_validation(
    X,
    y,
    model_fn,
    initial_window=756,
    step_size=5,
    forecast_horizon=1
):

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(X) - forecast_horizon, step_size):

        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i + forecast_horizon]
        y_test = y.iloc[i:i + forecast_horizon]

        # ------------------------
        # SCALE
        # ------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------------
        # TRAIN MODEL
        # ------------------------
        model = model_fn()
        model.fit(X_train_scaled, y_train)

        # ------------------------
        # PREDICT
        # ------------------------
        y_pred = model.predict(X_test_scaled)

        preds.extend(y_pred)
        actuals.extend(y_test.values)
        dates.extend(y_test.index)

    return np.array(preds), np.array(actuals), dates

def forecast_ml_next_day(model, df, feature_cols):
    last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    return model.predict(last_row)[0]