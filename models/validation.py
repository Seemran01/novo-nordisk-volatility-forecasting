from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pandas as pd


# =========================================================
# RMSE SCORER
# =========================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(
    rmse,
    greater_is_better=False
)


# =========================================================
# QLIKE SCORER
# =========================================================
def qlike_loss(y_true, y_pred):
    eps = 1e-8

    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)

    return np.mean(
        np.log(y_pred) + (y_true / y_pred)
    )


qlike_scorer = make_scorer(
    qlike_loss,
    greater_is_better=False
)


# =========================================================
# WALK-FORWARD VALIDATION 
# =========================================================
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

    inner_cv = TimeSeriesSplit(n_splits=3)

    for i in range(initial_window, len(X) - forecast_horizon, step_size):

        # ========================
        # OUTER SPLIT
        # ========================
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[i:i + forecast_horizon]
        y_test = y.iloc[i:i + forecast_horizon]

        # ========================
        # PIPELINE
        # ========================
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_fn())
        ])

        # ========================
        # GRID SEARCH (INNER CV)
        # ========================
        grid_search = GridSearchCV(
            estimator=pipe,   
            param_grid=param_grid,
            cv=inner_cv,
            scoring={
                "RMSE": rmse_scorer,
                "QLIKE": qlike_scorer
            },
            refit="RMSE",
            n_jobs=-1
        )

        # ========================
        # FIT (NO LEAKAGE)
        # ========================
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # ========================
        # PREDICT
        # ========================
        y_pred = best_model.predict(X_test)

        preds.extend(y_pred)
        actuals.extend(y_test.values)
        dates.extend(y_test.index)

    return np.array(preds), np.array(actuals), dates


# =========================================================
# NEXT-DAY FORECAST
# =========================================================
def forecast_ml_next_day(model, df, feature_cols):

    last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)

    return model.predict(last_row)[0]