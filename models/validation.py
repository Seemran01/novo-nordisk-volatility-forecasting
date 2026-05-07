from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd

def walk_forward_validation(
    X,
    y,
    model_fn,
    param_grid,
    initial_window=756,
    step_size=5,
    forecast_horizon=1
):

    preds, actuals, dates = [], [], []

    for i in range(initial_window, len(X) - forecast_horizon, step_size):

        # ========================
        # OUTER SPLIT
        # ========================
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i + forecast_horizon]
        y_test = y.iloc[i:i + forecast_horizon]

        # ========================
        # SCALE
        # ========================
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ========================
        # INNER CV LOOP
        # ========================
        inner_cv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            estimator=model_fn(),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_

        # ========================
        # PREDICT
        # ========================
        y_pred = best_model.predict(X_test_scaled)

        preds.extend(y_pred)
        actuals.extend(y_test.values)
        dates.extend(y_test.index)

    return np.array(preds), np.array(actuals), dates

def forecast_ml_next_day(model, df, feature_cols):
    last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    return model.predict(last_row)[0]