import numpy as np
import pandas as pd

from models.ml_models import rf_model, svr_model, xgb_model
from models.validation import walk_forward_validation
from utils.metrics import calculate_metrics
from data.data_pipeline import prepare_data

from models.econometric import (
    walk_forward_garch,
    walk_forward_har,
    walk_forward_ewma,
    walk_forward_naive
)


def run_all_models(df, selected_models, window_size, step_size):

    # =========================
    # DATA
    # =========================
    X, y = prepare_data(df)

    initial_window = min(window_size, len(X) - step_size * 5)

    model_results = {}

    # =========================
    # ML MODELS 
    # =========================
    ml_models = {
        "Random Forest": rf_model,
        "SVR": svr_model,
        "XGBoost": xgb_model
    }

    for name, model_fn in ml_models.items():
        if name in selected_models:

            pred, actual, dates = walk_forward_validation(
                X,
                y,
                model_fn,
                initial_window,
                step_size,
                forecast_horizon=22
            )

            if len(pred) > 0:
                model_results[name] = {
                    "pred": np.array(pred),
                    "actual": np.array(actual),
                    "dates": pd.to_datetime(dates),
                    **calculate_metrics(actual, pred)
                }

    # =========================
    # GARCH
    # =========================
    if "GARCH(1,1)" in selected_models:

        pred, actual, dates = walk_forward_garch(
            df, initial_window, step_size, forecast_horizon=22
        )

        if len(pred) > 0:
            model_results["GARCH(1,1)"] = {
                "pred": np.array(pred),
                "actual": np.array(actual),
                "dates": pd.to_datetime(dates),
                **calculate_metrics(actual, pred)
            }

    # =========================
    # HAR
    # =========================
    if "HAR-RV" in selected_models:

        pred, actual, dates = walk_forward_har(
            df, initial_window, step_size, forecast_horizon=22
        )

        if len(pred) > 0:
            model_results["HAR-RV"] = {
                "pred": np.array(pred),
                "actual": np.array(actual),
                "dates": pd.to_datetime(dates),
                **calculate_metrics(actual, pred)
            }

    # =========================
    # EWMA
    # =========================
    if "EWMA Volatility" in selected_models:

        pred, actual, dates = walk_forward_ewma(
            df["Log_Returns"],
            initial_window,
            step_size,
            forecast_horizon=22
        )

        if len(pred) > 0:
            model_results["EWMA Volatility"] = {
                "pred": np.array(pred),
                "actual": np.array(actual),
                "dates": pd.to_datetime(dates),
                **calculate_metrics(actual, pred)
            }

    # =========================
    # NAIVE
    # =========================
    if "Naive Persistence" in selected_models:

        pred, actual, dates = walk_forward_naive(
            y,
            initial_window,
            step_size,
            forecast_horizon=22
        )

        if len(pred) > 0:
            model_results["Naive Persistence"] = {
                "pred": np.array(pred),
                "actual": np.array(actual),
                "dates": pd.to_datetime(dates),
                **calculate_metrics(actual, pred)
            }

    return model_results