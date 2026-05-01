# =========================
# APP SETUP
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from pathlib import Path
from data.data_loader import load_data
from data.data_pipeline import prepare_data
from features.feature_engineering import create_features
from models.validation import walk_forward_validation
from models.ml_models import rf_model, svr_model, xgb_model
from models.model_pipeline import run_all_models
from utils.metrics import calculate_metrics
from utils.dm_test import dm_test
from models.econometric import forecast_garch_next, forecast_har_next, forecast_naive_next, forecast_ewma_next
from models.validation import forecast_ml_next_day



def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NVO Volatility Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================
# SESSION STATE
# =========================

if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["Random Forest", "EWMA Volatility", "GARCH(1,1)"]

if "window_size" not in st.session_state:
    st.session_state.window_size = 756

if "step_size" not in st.session_state:
    st.session_state.step_size = 5

if "selected_models" not in st.session_state:
    st.session_state.selected_models = [
        "Random Forest",
        "EWMA Volatility",
        "GARCH(1,1)"
    ]


# =========================
# SIDEBAR
# =========================
# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚡ NVO Volatility Predictor Settings")

# --- CALLBACK FUNCTIONS ---
def set_ml_only():
    st.session_state.update({
        "selected_models": [
            "Random Forest", 
            "SVR", 
            "XGBoost"
        ]
    })

def set_econometrics_only():
    st.session_state.update({
        "selected_models": [
            "GARCH(1,1)", 
            "HAR-RV"
        ]
    })

def set_baselines_only():
    st.session_state.update({
        "selected_models": [
            "Naive Persistence",
            "EWMA Volatility"
        ]
    })


# --- MODEL SELECTION ---
with st.sidebar.expander("🤖 Model Selection", expanded=True):

    all_models = [
        "Random Forest",
        "SVR",
        "XGBoost",
        "GARCH(1,1)",
        "EWMA Volatility",
        "Naive Persistence",
        "HAR-RV"
    ]

    st.session_state.selected_models = st.multiselect(
        "Select models",
        options=all_models,
        default=st.session_state.selected_models
    )
    
    col1, col2, col3 = st.columns(3)
    col1.button("ML Only", on_click=set_ml_only)
    col2.button("Econometrics Only", on_click=set_econometrics_only)
    col3.button("Baselines Only", on_click=set_baselines_only)

# --- TIME PERIOD SELECTION ---
with st.sidebar.expander("📅 Data Period", expanded=False):

    time_periods = {
        "1 Year": "1y",
        "2 Years": "2y",
        "3 Years": "3y",
        "5 Years": "5y",
        "10 Years": "10y"
    }

    selected_period = st.selectbox(
        "Time Period",
        options=list(time_periods.keys()),
        index=3,
        key="selected_period"
    )


# =========================
# AUTO TRAINING WINDOW RULE
# =========================

train_map = {
    "1 Year": 0.5,   # optional (6 months)
    "2 Years": 1,
    "3 Years": 2,
    "5 Years": 3,
    "10 Years": 3
}

train_years = train_map[selected_period]

# converts years -> trading days
window_size = int(train_years * 252)

# keeps only step size user controlled
with st.sidebar.expander("🔬 Walk-Forward Validation", expanded=False):

    step_size = st.slider(
        "Step Size",
        min_value=1,
        max_value=22,
        value=5,
        step=1
    )

st.sidebar.write(
    f"Auto training window: {train_years} years ({window_size} days)"
)


def reset_app():
    st.session_state.selected_models = [
        "Random Forest",
        "EWMA Volatility",
        "GARCH(1,1)"
    ]
    st.session_state.window_size = 756
    st.session_state.step_size = 5
    st.rerun()
    


# --- QUICK ACTIONS ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")

col1, col2, col3 = st.sidebar.columns(3)

col1.button("🔄 Reset", on_click=reset_app)


# --- MINI SUMMARY ---
models = st.session_state.get("selected_models", [])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔹 Current Configuration")

if models:
    st.sidebar.markdown(f"**Models:** {', '.join(models)}")
else:
    st.sidebar.warning("No models selected")

st.sidebar.markdown(f"**Period:** {selected_period}")
st.sidebar.markdown(f"**Window:** {window_size}")
st.sidebar.markdown(f"**Step:** {step_size}")



# =========================
# TITLE
# =========================

# --- MAIN APP TITLE ---
st.title("📈 Novo Nordisk (NVO) Volatility Predictor")

st.markdown("""
This dashboard presents **short-horizon volatility forecasts for Novo Nordisk stock**
using both **machine learning models** (Random Forest, SVR, XGBoost) and
traditional **econometric models** (HAR, GARCH, EWMA, Naive Persistence).

The objective is to evaluate and compare predictive performance across model classes
under a realistic financial forecasting setting.
""")

with st.expander("See methodology"):
    st.markdown("""
    **Methodology Summary**

• **Data**: Novo Nordisk historical daily stock prices  
• **Target variable**: Realized volatility (rolling variance of log returns)  
• **Forecasting approach**: Walk-forward validation to simulate real-time prediction and avoid look-ahead bias  

• **Models**:
  - Random Forest Regression  
  - Support Vector Regression (SVR)  
  - XGBoost Regression  
  - HAR-RV model  
  - GARCH(1,1)  
  - EWMA volatility model  
  - Naive persistence benchmark  

• **Evaluation metrics**:
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - QLIKE (volatility-specific loss function)

The framework enables a direct comparison between machine learning approaches
and classical econometric volatility models under identical forecasting conditions.
""")

# =========================
# LOAD DATA
# =========================
df = load_data(period=time_periods[selected_period])
df = create_features(df)

X, y = prepare_data(df)

feature_cols = X.columns

rf = rf_model().fit(X, y)
svr = svr_model().fit(X, y)
xgb = xgb_model().fit(X, y)

har_X = df[['RV_1D', 'RV_5D', 'RV_22D']]
har_y = df['Realized_Vol']

har_model = LinearRegression()
har_model.fit(har_X, har_y)

st.write("X shape:", X.shape)

# =========================
# RUN ALL MODELS
# =========================

st.write("DEBUG selected_models:", st.session_state.selected_models)

model_results = run_all_models(
    df,
    st.session_state.selected_models,
    window_size,
    step_size
)

st.write("DEBUG model_results keys:", list(model_results.keys()))

for m, res in model_results.items():
    st.write(m)
    st.write("pred:", len(res["pred"]))
    st.write("actual:", len(res["actual"]))
    st.write("dates:", len(res["dates"]))
    st.write("---")

if len(model_results) == 0:
    st.warning("No models selected or no results generated.")
    st.stop()


# =========================
# STRONG ALIGNMENT 
# =========================

valid_models = model_results

all_dates = []

for res in model_results.values():
    if "dates" in res and len(res["dates"]) > 0:
        all_dates.append(res["dates"])

common_start = max(d[0] for d in all_dates)
common_end = min(d[-1] for d in all_dates)

st.write("VALID MODELS:", list(valid_models.keys()))

if len(all_dates) == 0:
    st.error("No valid dates found")
    st.stop()

first_key = list(valid_models.keys())[0]

common_index = set(valid_models[first_key]["dates"])

for res in valid_models.values():
    common_index = common_index.intersection(set(res["dates"]))

common_index = sorted(list(common_index))


# =========================
# STORE FOR DM TEST
# =========================
st.session_state.results = {
    m: res["pred"] for m, res in model_results.items()
}

st.session_state.actuals = list(model_results.values())[0]["actual"]


# =========================
# METRICS DISPLAY
# =========================

metrics_df = pd.DataFrame({
    m: {
        "MAE": res["MAE"],
        "RMSE": res["RMSE"],
        "QLIKE": res["QLIKE"]
    }
    for m, res in model_results.items()
}).T


# =========================
# HEATMAP
# =========================
fig = px.imshow(
    metrics_df,
    text_auto=".4f",
    aspect="auto",
    color_continuous_scale="RdYlGn_r",
    zmin=metrics_df.min().min(),
    zmax=metrics_df.max().max(),
)

fig.update_layout(
    title="Model Performance Heatmap",
    title_x=0.5,
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="#e5e7eb"),
)

fig.update_traces(
    textfont_size=12
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# FUTURE FORECAST 
# =========================

future_results = {}

selected = st.session_state.selected_models
st.write("Selected models:", selected)

if "GARCH(1,1)" in selected:
    future_results["GARCH(1,1)"] = forecast_garch_next(df)

if "HAR-RV" in selected:
    future_results["HAR-RV"] = forecast_har_next(har_model, df)

if "EWMA Volatility" in selected:
    future_results["EWMA"] = forecast_ewma_next(df["Log_Returns"])

if "Naive Persistence" in selected:
    future_results["Naive"] = forecast_naive_next(df)

if "Random Forest" in selected:
    future_results["Random Forest"] = forecast_ml_next_day(rf, df, feature_cols)

if "SVR" in selected:
    future_results["SVR"] = forecast_ml_next_day(svr, df, feature_cols)

if "XGBoost" in selected:
    future_results["XGBoost"] = forecast_ml_next_day(xgb, df, feature_cols)


# DEBUG
st.write("DEBUG forecasts:")
for k, v in future_results.items():
    st.write(k, v)

# =========================
# DISPLAY
# =========================

st.subheader("🔮 Next-Day Volatility Forecast")

st.dataframe(
    pd.DataFrame.from_dict(future_results, orient="index", columns=["Forecast"])
    .reset_index()
    .rename(columns={"index": "Model"})
)

# =========================
# ACTUAL VOL
# =========================
st.subheader("📌 Latest Realized Volatility")
st.metric("Latest RV", f"{y.iloc[-1]:.6f}")


# =========================
# TABLE 
# =========================
st.subheader("📌 Latest Volatility: Actual vs Model Predictions")

if len(model_results) == 0:
    st.warning("No model results available.")
    st.stop()

rows = []

# shared actuals
actuals = next(iter(model_results.values()))["actual"]

last_actual = actuals.iloc[-1] if isinstance(actuals, pd.Series) else actuals[-1]

rows.append({
    "Model": "Actual Volatility",
    "Value": last_actual,
    "Type": "Actual"
})

for model_name, res in model_results.items():
    rows.append({
        "Model": model_name,
        "Value": res["pred"][-1],
        "Type": "Prediction"
    })

df = pd.DataFrame(rows)

# =========================
# STYLING 
# =========================
styled_df = (
    df.style
    .format({"Value": "{:.6f}"})
    .apply(lambda x: ["background-color: #1e293b" if v == "Prediction" else "background-color: #0f172a"
                      for v in df["Type"]], axis=0)
    .set_properties(**{
        "color": "#e5e7eb",
        "text-align": "center"
    })
)

st.dataframe(styled_df, use_container_width=True)


# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("🧠 Feature Importance (ML Models)")

# --- Ensure flat columns (fix MultiIndex issue) ---
if isinstance(X.columns, pd.MultiIndex):
    X.columns = ['_'.join(map(str, col)) for col in X.columns]

feature_cols = X.columns

# RANDOM FOREST
if "Random Forest" in st.session_state.selected_models:
    try:
        importance = pd.Series(
            rf.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)

        st.write("Random Forest Feature Importance")
        st.bar_chart(importance)

    except Exception as e:
        st.warning(f"RF importance error: {e}")

# XGBOOST
if "XGBoost" in st.session_state.selected_models:
    try:
        importance = pd.Series(
            xgb.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)

        st.write("XGBoost Feature Importance")
        st.bar_chart(importance)

    except Exception as e:
        st.warning(f"XGBoost importance error: {e}")

# SVR
if "SVR" in st.session_state.selected_models:
    st.info("SVR does not provide built-in feature importance.")


# =========================
# DM TEST
# =========================
st.subheader("📉 Diebold–Mariano Test")

base_model = next(iter(model_results.values()))
actuals = base_model["actual"]
models = st.session_state.results

model_names = list(model_results.keys())
dm_results = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):

        m1, m2 = model_names[i], model_names[j]

        pred1 = models[m1]
        pred2 = models[m2]

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

st.dataframe(pd.DataFrame(dm_results))


# =========================
# ALIGN ALL MODELS TO COMMON DATE RANGE
# =========================

# gets all available date arrays safely
all_dates = [res.get("dates") for res in model_results.values() if "dates" in res]

# removes None / empty
all_dates = [d for d in all_dates if d is not None and len(d) > 0]

if len(all_dates) > 0:
    common_start = max(d[0] for d in all_dates)
    common_end = min(d[-1] for d in all_dates)

    # filters each model to same time window
    for m in model_results:
        if "dates" in model_results[m]:
            dates = pd.Index(model_results[m]["dates"])

            mask = (dates >= common_start) & (dates <= common_end)

            model_results[m]["pred"] = np.array(model_results[m]["pred"])[mask]
            model_results[m]["actual"] = np.array(model_results[m]["actual"])[mask]
            model_results[m]["dates"] = dates[mask]


# =========================
# DM TEST
# =========================

st.subheader("📈 Actual vs Predicted Volatility")

plot_frames = []

# =========================
# PREDICTIONS
# =========================
for model_name, res in model_results.items():

    n = min(
        len(res["pred"]),
        len(res["actual"]),
        len(res["dates"])
    )

    if n == 0:
        continue

    pred_df = pd.DataFrame({
        "Date": res["dates"][:n],
        "Volatility": res["pred"][:n],
        "Model": model_name
    })

    plot_frames.append(pred_df)


# =========================
# ACTUAL SERIES 
# =========================
first_model = next(iter(model_results.values()))

n_actual = min(
    len(first_model["actual"]),
    len(first_model["dates"])
)

actual_df = pd.DataFrame({
    "Date": first_model["dates"][:n_actual],
    "Volatility": first_model["actual"][:n_actual],
    "Model": "Actual"
})

plot_frames.append(actual_df)


# =========================
# BEST MODEL SELECTION
# =========================

if len(model_results) > 0:

    best_model = min(
        model_results.items(),
        key=lambda x: x[1]["QLIKE"]  
    )

    best_model_name = best_model[0]
    best_qlike = best_model[1]["QLIKE"]

    st.success(f"🏆 Best Model: {best_model_name} (QLIKE: {best_qlike:.6f})")

ranking = sorted(
    [(m, res["QLIKE"]) for m, res in model_results.items()],
    key=lambda x: x[1]
)

ranking_df = pd.DataFrame(ranking, columns=["Model", "QLIKE"])

st.subheader("📊 Model Ranking (Best → Worst)")
st.dataframe(ranking_df)



# =========================
# PLOT
# =========================
plot_df = pd.concat(plot_frames)

plot_df = (
    plot_df
    .groupby(["Date","Model"], as_index=False)
    .mean()
)

fig = px.line(
    plot_df,
    x="Date",
    y="Volatility",
    color="Model",
    template="plotly_white",
    title=f"Actual vs Predicted Volatility (Best: {best_model_name})"
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# DOWNLOAD
# =========================
csv = (
    plot_df
    .groupby(["Date","Model"], as_index=False)
    .mean()
    .pivot(
        index="Date",
        columns="Model",
        values="Volatility"
    )
    .reset_index()
    .to_csv(index=False)
)

st.download_button(
    "📥 Download Predictions",
    csv,
    "volatility_predictions.csv",
    "text/csv"
)