# --- APP SETUP ---
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

from data.data_loader import load_data
from features.feature_engineering import create_features
from models.validation import walk_forward_validation
from utils.dm_test import dm_test
from models.ml_models import rf_model, svr_model, xgb_model
from models.econometric import run_garch, run_har, naive_persistence, ewma_volatility
from utils.metrics import calculate_metrics

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NVO Volatility Predictor",
    page_icon="📈",
    layout="wide"
)

# --- LOAD CSS ---
try:
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css not found. Default styling will be used.")


# Initialising session
if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["Random Forest", "HAR-RV"]

if "results" not in st.session_state:
    st.session_state.results = {}

if "actuals" not in st.session_state:
    st.session_state.actuals = None

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚡ NVO Volatility Predictor Settings")

# --- SESSION STATE DEFAULTS ---
if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["Random Forest", "XGBoost", "GARCH(1,1)"]

# --- CALLBACK FUNCTIONS ---
def set_ml_only():
    st.session_state.selected_models = ["Random Forest", "SVR", "XGBoost"]

def set_econometrics_only():
    st.session_state.selected_models = ["GARCH(1,1)", "HAR-RV"]

def set_baselines_only():
    st.session_state.selected_models = ["Naive Persistence", "EWMA Volatility"]


# --- MODEL SELECTION ---
with st.sidebar.expander("📊 Model Selection", expanded=True):
    st.markdown("**Select which models to include:**")
    
    selected_models = st.multiselect(
        "Choose Models",
        options=["Random Forest", "SVR", "XGBoost", "GARCH(1,1)", "HAR-RV", "Naive Persistence", "EWMA Volatility"],
        key="selected_models",
        help="Machine Learning models (RF, SVR, XGBoost), Econometric benchmarks (GARCH, HAR-RV), and Baselines (Naive, EWMA)"
    )
    
    col1, col2, col3 = st.columns(3)
    col1.button("ML Only", on_click=set_ml_only)
    col2.button("Econometrics Only", on_click=set_econometrics_only)
    col3.button("Baselines Only", on_click=set_baselines_only)

# --- TIME PERIOD SELECTION ---
with st.sidebar.expander("📅 Data Period", expanded=False):
    st.markdown("**Select historical data range:**")
    time_periods = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y"
    }
    selected_period = st.selectbox(
        "Time Period",
        options=list(time_periods.keys()),
        index=3,
        help="Select length of historical data for features"
    )
    col1, col2 = st.columns(2)
    if col1.button("1 Month Shortcut"):
        selected_period = "1 Month"
    if col2.button("1 Year Shortcut"):
        selected_period = "1 Year"

# --- POPUP WARNING FOR SHORT PERIODS ---
    short_periods = ["1 Month", "3 Months", "6 Months"]
    if selected_period in short_periods:
        st.warning(
            f"⚠️ Warning: {selected_period} is a short period. "
            "Predictions may be unreliable. Use 1 year or more for better results."
        )

# --- WALK-FORWARD VALIDATION ---
with st.sidebar.expander("🔬 Walk-Forward Validation", expanded=False):
    st.markdown("**Set window and step size:**")
    window_size = st.slider("Window Size", min_value=50, max_value=200, value=100, step=10,
                            help="Past days used for training in each step")
    step_size = st.slider("Step Size", min_value=5, max_value=50, value=10, step=5,
                          help="Days to move forward after each iteration")

# --- QUICK ACTIONS ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("🔄 Reset"):
    st.session_state.selected_models = ["Random Forest", "XGBoost", "GARCH(1,1)"]
    selected_period = "1 Year"
    window_size = 100
    step_size = 10

# --- MINI SUMMARY ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔹 Current Configuration")
st.sidebar.markdown(f"**Models:** {', '.join(st.session_state.selected_models)}")
st.sidebar.markdown(f"**Period:** {selected_period}")
st.sidebar.markdown(f"**Walk-Forward:** Window={window_size}, Step={step_size}")

# --- MAIN APP TITLE ---
st.title("📈 Novo Nordisk (NVO) Volatility Predictor")

st.markdown("""
This dashboard presents **short-horizon volatility forecasts for Novo Nordisk stock**
using **machine learning models** (Random Forest, SVR, XGBoost).

Model performance is **benchmarked against traditional econometric models**
(HAR and GARCH) to evaluate predictive accuracy in a realistic setting.

Evaluation is conducted using **walk-forward validation**, simulating real-time forecasting.
""")

with st.expander("See methodology"):
    st.markdown("""
    **Methodology Summary**

• Data: Novo Nordisk historical stock prices  
• Target variable: Realized volatility (rolling squared log returns)  
• Validation: Walk-forward validation to simulate real-time forecasting and avoid look-ahead bias. 
• Models:
  - Random Forest
  - Support Vector Regression
  - XGBoost
  - HAR
  - GARCH(1,1)

The objective is to compare machine learning methods with traditional
econometric benchmarks for short-horizon volatility forecasting.

• Metrics:
  - MAE
  - RMSE
  - MAPE
  - R2             
""")


# --- LOAD DATA ---
df = load_data(period=time_periods[selected_period])
if df is None or len(df) < 5:
    st.warning(f"⚠️ Not enough data for {selected_period}. Please choose a longer period.")
    st.stop()

# --- FEATURE ENGINEERING ---
df = create_features(df)

# --- WARNING FOR SHORT PERIODS ---
short_periods = ["1 Month", "3 Months", "6 Months"]
if df.empty:
    if selected_period in short_periods:
        st.warning(
            f"⚠️ After feature engineering, no usable data for {selected_period}. "
            "Select 1 year or more for meaningful results."
        )
    else:
        st.warning("⚠️ No data available after feature engineering. Please select a longer period.")
    st.stop()

# --- SELECT FEATURES AND TARGET ---
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_5', 'MA_20', 'Volatility_10', 'Volume_Change'
]

X = df[feature_cols]
y = df['Realized_Vol']

if len(X) == 0 or len(y) == 0:
    st.warning("⚠️ Feature or target arrays are empty. Cannot proceed.")
    st.stop()

# --- READY FOR MODELS ---
model_results = {}

# --- ML MODELS ---
st.write("X length:", len(X))
st.write("window_size:", window_size)
st.write("step_size:", step_size)

if "Random Forest" in st.session_state.selected_models:
    X = X.ffill().bfill()
    rf_pred, rf_actual, rf_dates = walk_forward_validation(X, y, rf_model, window_size, step_size)
    if len(rf_pred) == 0 or len(rf_actual) == 0:
        st.warning("⚠️ Random Forest could not generate predictions. Data may be too short.")
    else:
        rf_metrics = calculate_metrics(rf_actual, rf_pred)
        model_results["Random Forest"] = {"pred": rf_pred, "actual": rf_actual, "dates": rf_dates, **rf_metrics}

if "SVR" in st.session_state.selected_models:
    svr_pred, svr_actual, svr_dates = walk_forward_validation(X, y, svr_model, window_size, step_size)
    if len(svr_pred) == 0 or len(svr_actual) == 0:
        st.warning("⚠️ SVR could not generate predictions. Data may be too short.")
    else:
        svr_metrics = calculate_metrics(svr_actual, svr_pred)
        model_results["SVR"] = {"pred": svr_pred, "actual": svr_actual, "dates": svr_dates, **svr_metrics}

if "XGBoost" in st.session_state.selected_models:
    xgb_pred, xgb_actual, xgb_dates = walk_forward_validation(X, y, xgb_model, window_size, step_size)
    if len(xgb_pred) == 0 or len(xgb_actual) == 0:
        st.warning("⚠️ XGBoost could not generate predictions. Data may be too short.")
    else:
        xgb_metrics = calculate_metrics(xgb_actual, xgb_pred)
        model_results["XGBoost"] = {"pred": xgb_pred, "actual": xgb_actual, "dates": xgb_dates, **xgb_metrics}

# --- ECONOMETRIC MODELS ---
if "GARCH(1,1)" in st.session_state.selected_models:
    garch_pred, garch_dates = run_garch(df)
    if len(garch_pred) == 0:
        st.warning("⚠️ GARCH could not generate predictions. Data may be too short.")
    else:
        garch_metrics = calculate_metrics(y[-len(garch_pred):], garch_pred)
        model_results["GARCH(1,1)"] = {"pred": garch_pred, "actual": y[-len(garch_pred):], "dates": garch_dates, **garch_metrics}

if "HAR-RV" in st.session_state.selected_models:
    # In app.py, right before run_har(df):
    st.write("RV_1D sample:", df['RV_1D'].dropna().tail(10).values)
    st.write("Realized_Vol sample:", df['Realized_Vol'].dropna().tail(10).values)
    st.write("df shape:", df.shape)

    har_pred, har_actual, har_dates = run_har(df, test_size=0.2)   

    if len(har_pred) == 0:
        st.warning("⚠️ HAR-RV could not generate predictions. Data may be too short.")
    else:
        har_metrics = calculate_metrics(har_actual, har_pred)

        model_results["HAR-RV"] = {
            "pred": har_pred,
            "actual": har_actual,
            "dates": har_dates,
            **har_metrics
        }
# --- BaseLines Only
if "Naive Persistence" in st.session_state.selected_models:
    naive_pred = naive_persistence(df['Realized_Vol'])
    
    naive_metrics = calculate_metrics(y[1:], naive_pred.dropna().values)

    model_results["Naive Persistence"] = {
        "pred": naive_pred.dropna().values,
        "actual": y[1:],
        "dates": df.index[1:],
        **naive_metrics
    }

if "EWMA Volatility" in st.session_state.selected_models:
    ewma_pred, ewma_actual, ewma_dates = ewma_volatility(df['Log_Returns'])

    ewma_metrics = calculate_metrics(ewma_actual, ewma_pred)

    model_results["EWMA"] = {
        "pred": ewma_pred,
        "actual": ewma_actual,
        "dates": ewma_dates,
        **ewma_metrics
    }

# --- STOP IF NO MODELS PRODUCED RESULTS ---
if len(model_results) == 0:
    st.warning("⚠️ No models could generate predictions. Increase data period or reduce walk-forward window size.")
    st.stop()

st.session_state.actuals = list(model_results.values())[0]["actual"]

first_model = next(iter(model_results.values()))

st.session_state.results = {
    m: res["pred"] for m, res in model_results.items()
}

st.session_state.actuals = first_model["actual"]

# --- ENHANCED MODEL METRICS DISPLAY ---
import plotly.graph_objects as go

st.subheader("📊 Model Metrics Dashboard")

# Colors for metrics
metric_colors = {
    "MAE": "#FF6B6B",      # Red
    "RMSE": "#4ECDC4",     # Teal
    "R²": "#FFD93D",       # Yellow
    "MAPE": "#21B0C6"      # Dark teal
}

# Create side-by-side columns for each model
model_cols = st.columns(len(model_results))

for i, (model, res) in enumerate(model_results.items()):
    with model_cols[i]:
        # Card container
        card_html = f"""
        <div style="
            background-color:#2C3E50;
            padding:20px;
            border-radius:15px;
            border:1px solid #ddd;
            text-align:center;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="color:#1f77b4;margin-bottom:15px;">{model}</h3>
            <p style="font-size:16px;"><strong style='color:{metric_colors['MAE']}'>MAE:</strong> {res['MAE']:.6f}</p>
            <p style="font-size:16px;"><strong style='color:{metric_colors['RMSE']}'>RMSE:</strong> {res['RMSE']:.6f}</p>
            <p style="font-size:16px;"><strong style='color:{metric_colors['R²']}'>R²:</strong> {res['R2']:.4f}</p>
            <p style="font-size:16px;"><strong style='color:{metric_colors['MAPE']}'>MAPE:</strong> {res['MAPE']:.2f}%</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Mini bar chart for metrics comparison
        fig = go.Figure(go.Bar(
            x=["MAE", "RMSE", "R²", "MAPE"],
            y=[res['MAE'], res['RMSE'], res['R2'], res['MAPE']],
            marker_color=[metric_colors[m] for m in ["MAE","RMSE","R²","MAPE"]],
            text=[f"{v:.4f}" if m != "MAPE" else f"{v:.2f}%" for m,v in zip(["MAE","RMSE","R²","MAPE"], [res['MAE'], res['RMSE'], res['R2'], res['MAPE']])],
            textposition="auto"
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(title="Value"),
            xaxis=dict(title="Metric")
        )
        st.plotly_chart(fig, use_container_width=True)

# Build plot using dates per model
fig = go.Figure()

for model_name, res in model_results.items():
    fig.add_trace(go.Scatter(
        x=res['dates'],
        y=res['pred'],
        mode='lines',
        name=model_name
    ))
    # Plot actual once (use HAR or any model's actual + dates)

# Add actual from any model (they should share the same test period)
first = next(iter(model_results.values()))
fig.add_trace(go.Scatter(
    x=first['dates'],
    y=first['actual'],
    mode='lines',
    name='Actual',
    line=dict(color='white', dash='dash')
))

fig.update_layout(
    title="Actual vs Predicted Volatility",
    xaxis_title="Date",
    yaxis_title="Volatility",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

#Realized Volatility

st.subheader("📌 Current Realized Volatility")

latest_rv = y.iloc[-1]
st.metric(
    label="Latest Realized Volatility",
    value=f"{latest_rv:.6f}"
)

#Table
st.subheader("📌 Latest Volatility: Actual vs Model Predictions")

if len(model_results) == 0:
    st.warning("No model results available.")
    st.stop()

rows = []

# get shared actuals
actuals = next(iter(model_results.values()))["actual"]

# handle both pandas Series and numpy arrays safely
if isinstance(actuals, pd.Series):
    last_actual = actuals.iloc[-1]
else:
    last_actual = actuals[-1]

rows.append({
    "Model": "Actual Volatility",
    "Value": last_actual
})

for model_name, res in model_results.items():
    rows.append({
        "Model": model_name,
        "Value": res["pred"][-1]
    })

df = pd.DataFrame(rows)
df["Value"] = df["Value"].apply(lambda x: f"{x:.6f}")

st.dataframe(df)

#Feature Importance
st.subheader("🧠 Feature Importance (ML Models)")

feature_names = X.columns

# RANDOM FOREST
# CLEANING FEATURES
if isinstance(X.columns, pd.MultiIndex):
    X.columns = ['_'.join(col).strip() for col in X.columns]

X = X.loc[:, ~X.columns.str.contains("index", case=False)]
if "Random Forest" in st.session_state.selected_models:
    try:
        model = rf_model()
        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        st.write("Random Forest Feature Importance")
        st.bar_chart(importance)

    except Exception as e:
        st.warning(f"Feature importance unavailable for Random Forest: {e}")

# XGBOOST
# CLEANING FEATURES
if isinstance(X.columns, pd.MultiIndex):
    X.columns = ['_'.join(col).strip() for col in X.columns]

X = X.loc[:, ~X.columns.str.contains("index", case=False)]
if "XGBoost" in st.session_state.selected_models:
    try:
        model = xgb_model()
        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        st.write("XGBoost Feature Importance")
        st.bar_chart(importance)

    except Exception as e:
        st.warning(f"Feature importance unavailable for XGBoost: {e}")

# SVR (no feature importance)
if "SVR" in st.session_state.selected_models:
    st.info("SVR does not provide built-in feature importance.") 

# ---------------------------
# DM TEST
# ---------------------------
def dm_test(actuals, pred1, pred2):
    d = (actuals - pred1)**2 - (actuals - pred2)**2
    dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


st.subheader("📉 Diebold–Mariano Test")

# Check data exists
if "results" not in st.session_state or "actuals" not in st.session_state:
    st.warning("No results found. Run the models first.")
    st.stop()

actuals = st.session_state.actuals
models = st.session_state.results

selected_models = st.session_state.get("selected_models", [])

model_names = [m for m in models.keys() if m in selected_models]
dm_results = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):

        m1 = model_names[i]
        m2 = model_names[j]

        pred1 = models[m1]
        pred2 = models[m2]

        # Align lengths just in case
        min_len = min(len(actuals), len(pred1), len(pred2))

        dm_stat, p_val = dm_test(
            actuals[-min_len:],
            pred1[-min_len:],
            pred2[-min_len:]
        )

        dm_results.append({
            "Model 1": m1,
            "Model 2": m2,
            "DM Stat": round(dm_stat, 3),
            "p-value": round(p_val, 3),
            "Significant": "Yes ✅" if p_val < 0.05 else "No ❌"
        })

df_dm = pd.DataFrame(dm_results)
st.dataframe(df_dm)


# --- COMPARISON TABLE ---
st.subheader("📊 Model Metrics Comparison")
metrics_df = pd.DataFrame({m: {"MAE":res['MAE'],"RMSE":res['RMSE'],"R2":res['R2'],"MAPE":res['MAPE']} for m,res in model_results.items()}).T.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
metrics_df.rename(columns={'index':'Model'}, inplace=True)
fig2 = px.bar(metrics_df, x='Model', y='Value', color='Metric', barmode='group', text='Value', template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# --- PLOTS ---
st.subheader("📈 Actual vs Predicted Volatility")

# Find the minimum length among all predictions + actuals
min_len = min(len(actuals), *(len(res['pred']) for res in model_results.values()))

# Slice actuals and all predictions to this length
aligned_actuals = actuals[-min_len:]
plot_dict = {m: res['pred'][-min_len:] for m,res in model_results.items()}
plot_dict['Actual'] = aligned_actuals

plot_df = pd.DataFrame(plot_dict)

# Melt for Plotly
plot_df = plot_df.reset_index().rename(columns={'index': 'Date'})

plot_df = plot_df.melt(
    id_vars='Date',
    var_name='Model',
    value_name='Volatility'
)

fig = px.line(
    plot_df,
    x='Date',
    y='Volatility',
    color='Model',
    title="Actual vs Predicted Volatility",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --- Download Predictions ---
date_col = 'Date' if 'Date' in plot_df.columns else plot_df.columns[0]

csv_data = plot_df.pivot(
    index=date_col,
    columns='Model',
    values='Volatility'
).reset_index().to_csv(index=False)

st.download_button(
    "📥 Download Predictions",
    csv_data,
    "nvo_volatility_predictions.csv",
    "text/csv"
)


