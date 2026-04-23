import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

os.makedirs("models/saved_models", exist_ok=True)

# Download data
data = yf.download("NOVO-B.CO", start="2015-01-01", end="2025-01-01")
data = data[['Close']]
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Volatility'] = data['Return'].rolling(10).std() * np.sqrt(252)

# Feature engineering
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_20'] = data['Close'].rolling(20).mean()
data['Lag1_Return'] = data['Return'].shift(1)
data['Lag2_Return'] = data['Return'].shift(2)
data.dropna(inplace=True)

X = data[['MA_5', 'MA_20', 'Lag1_Return', 'Lag2_Return']]
y = data['Volatility']

# Train RF
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
rf_model.fit(X, y)

# Save model
joblib.dump(rf_model, "models/saved_models/random_forest.pkl")
print("Random Forest model trained and saved!")