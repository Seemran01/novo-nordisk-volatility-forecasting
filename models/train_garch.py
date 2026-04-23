from arch import arch_model
from utils.preprocessing import download_data, compute_returns
import joblib

data = download_data()
data = compute_returns(data)

returns = data["Returns"] * 100  # arch prefers scaled data

model = arch_model(returns, vol="Garch", p=1, q=1)
res = model.fit(disp="off")

joblib.dump(res, "models/saved_models/garch_model.pkl")

print("GARCH model trained and saved!")