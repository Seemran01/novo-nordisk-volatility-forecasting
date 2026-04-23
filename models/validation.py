import numpy as np
import pandas as pd

def walk_forward_validation(X, y, model_func, window_size=100, step_size=10):

    predictions, actuals, dates = [], [], []

    for i in range(window_size, len(X), step_size):

        X_train = X.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]

        X_test = X.iloc[i:i+1]
        y_test = y.iloc[i:i+1]

        if len(X_test) == 0:
            break

        model = model_func()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        predictions.append(pred)
        actuals.append(y_test.iloc[0])
        dates.append(X.index[i])

    return np.array(predictions), np.array(actuals), pd.DatetimeIndex(dates)