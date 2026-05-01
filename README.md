# 📊 Volatility Forecasting System (Novo Nordisk)

This project implements a **financial volatility forecasting framework** combining **econometric models** and **machine learning models** to predict and evaluate stock market volatility.

The system is designed for **short-horizon forecasting** using a realistic **walk-forward validation framework**, ensuring no look-ahead bias.

---

## 🚀 Features

- 📉 Econometric models: GARCH(1,1), HAR-RV  
- 🤖 Machine Learning models: Random Forest, SVR, XGBoost  
- 📊 Benchmark models: Naive Persistence, EWMA  
- 🔁 Walk-forward validation for realistic time-series evaluation  
- 📈 Interactive Streamlit dashboard for live model comparison  
- 📊 Forecast evaluation using volatility-specific metrics (QLIKE, MAE, RMSE)  
- 📉 Visual comparison of actual vs predicted volatility  

---

## 🧠 Models Used

### Econometric Models
- GARCH(1,1)
- HAR-RV (Heterogeneous Autoregressive model)
- EWMA (Exponentially Weighted Moving Average)

### Machine Learning Models
- Random Forest Regression  
- Support Vector Regression (SVR)  
- XGBoost Regression  

### Baseline Models
- Naive Persistence (random walk benchmark)

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- QLIKE (preferred volatility loss function for variance forecasting)

---

## 🛠️ Methodology

- **Data**: Daily stock prices of Novo Nordisk (NVO)  
- **Target variable**: Realized volatility (variance of log returns)  
- **Validation**: Walk-forward (rolling/expanding window forecasting)  
- **Forecast horizon**: Short-term multi-step ahead predictions  
- **Objective**: Compare machine learning models with econometric benchmarks under identical forecasting conditions  

---

## 🧪 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- ARCH library (GARCH model)  
- Streamlit  
- Plotly  

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py