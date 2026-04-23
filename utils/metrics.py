from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def calculate_metrics(actual,pred):

    mae=mean_absolute_error(actual,pred)
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mape = (abs((actual - pred)/actual).mean()) * 100
    r2=r2_score(actual,pred)

    return {
        "MAE":mae,
        "RMSE":rmse,
        "R2":r2,
        "MAPE": mape
    }