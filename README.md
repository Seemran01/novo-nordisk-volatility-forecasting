# # Volatility Forecasting System (Novo Nordisk)

This project implements a financial volatility forecasting framework combining econometric and machine learning models to predict and evaluate stock market volatility.

The system is designed for short-horizon forecasting using a walk-forward validation framework to ensure realistic out-of-sample evaluation and prevent look-ahead bias.

## Features

- Econometric models: GARCH(1,1), HAR-RV  
- Machine learning models: Random Forest, Support Vector Regression (SVR), XGBoost  
- Benchmark models: Naive Persistence, EWMA  
- Walk-forward validation for time-series evaluation  
- Interactive Streamlit dashboard for model comparison  
- Volatility-specific evaluation metrics (QLIKE, MAE, RMSE)  
- Visual comparison of actual vs predicted volatility  

## Models Used

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
- EWMA Volatility  

## Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- QLIKE (quasi-likelihood loss function for volatility forecasting)  

## Methodology

- Data: Daily stock prices of Novo Nordisk (NVO)  
- Target variable: Realized volatility computed from squared log returns  
- Validation: Expanding window walk-forward validation  
- Forecast horizon: One-step-ahead daily volatility prediction  
- Objective: Compare machine learning models against econometric benchmarks under identical forecasting conditions  

## Tech Stack

- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- ARCH (for GARCH models)  
- Streamlit  
- Plotly  

## Installation and Setup

Clone the repository:
git clone <repository-url>
cd <project-folder>

Create a virtual environment:
python -m venv venv

Activate the virtual environment:

Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

## Notes

- Ensure all dependencies are installed inside the virtual environment  
- Walk-forward validation increases computation time  
- All models are evaluated under identical forecasting conditions for fair comparison  