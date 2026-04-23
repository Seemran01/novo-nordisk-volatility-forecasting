from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

def rf_model():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

def svr_model():
    return SVR(kernel="rbf")

def xgb_model():
    return xgb.XGBRegressor(
        n_estimators=100,
        random_state=42
    )