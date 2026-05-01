import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, pred):

    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)

    print("DEBUG actual min/max:", np.min(actual), np.max(actual))
    print("DEBUG pred min/max:", np.min(pred), np.max(pred))

    # =========================
    # REMOVE INVALID VALUES ONCE
    # =========================
    epsilon = 1e-8

    actual = np.clip(actual, epsilon, None)
    pred = np.clip(pred, epsilon, None)

    # =========================
    # CORE METRICS
    # =========================
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    # =========================
    # QLIKE (standard volatility loss)
    # =========================
    qlike = np.mean(
        (actual / pred) - np.log(actual / pred) - 1
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "QLIKE": qlike
    }