import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ForecastIQ - Time Series Forecasting Platform",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #8A8F98;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background-color: #171B26;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2B303C;
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #00F2FE;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8A8F98;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint definition
API_URL = os.getenv("FORECASTIQ_API_URL", "http://localhost:8002")

# Mocks in case server is loading weights
def mock_predict(history, model_type):
    # Sinusoidal extrapolation + minor trend
    x = np.arange(24)
    last_val = history[-1]
    trend = 0.5 * x
    seasonal = 10 * np.sin(x / 4)
    forecast = (last_val + trend + seasonal).tolist()
    return forecast

def mock_backtest(model_type):
    time = np.arange(120)
    vals = 100 + 20 * np.sin(time / 12) + 0.1 * time
    history = vals[:96].tolist()
    actuals = vals[96:].tolist()
    # Add noise to forecast
    np.random.seed(42)
    forecast = (vals[96:] + np.random.normal(0, 3, 24)).tolist()
    
    mae = float(np.mean(np.abs(np.array(actuals) - np.array(forecast))))
    rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(forecast)) ** 2)))
    
    return {
        "model_type": model_type,
        "history": history,
        "actuals": actuals,
        "forecast": forecast,
        "metrics": {"MAE": mae, "RMSE": rmse}
    }

# Header Section
col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<div class='main-title'>ForecastIQ</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Modern Time-Series Forecasting Platform (PatchTST vs N-BEATS vs TFT)</div>", unsafe_allow_html=True)
with col2:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            st.success("API: Connected")
        else:
            st.warning("API: Initializing")
    except Exception:
        st.error("API: Offline (Using Fallback)")

# Tabs
tab1, tab2 = st.tabs(["📊 Forecasting Console", "📈 Backtesting & Model Benchmarks"])

# TAB 1: FORECASTING CONSOLE
with tab1:
    st.markdown("### Interactive Multi-Horizon Predictor")
    col_fc_ctrl, col_fc_chart = st.columns([3, 7])
    
    with col_fc_ctrl:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Input Signal Configuration")
        signal_preset = st.selectbox("Choose Input Preset Trend", [
            "Sine Wave with Trend", 
            "Upward Linear Drift", 
            "Seasonal Demand Cycle"
        ])
        model_choice = st.selectbox("Forecast Model Architecture", ["PatchTST", "N-BEATS", "TFT"])
        noise_level = st.slider("Historical Noise Scale", 0.0, 5.0, 1.5, 0.5)
        st.markdown("</div>", unsafe_allow_html=True)
        
        btn_forecast = st.button("⚡ Generate 24-Step Forecast", use_container_width=True)
        
    with col_fc_chart:
        # Generate signal based on choice
        time = np.arange(96)
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, 96)
        
        if signal_preset == "Sine Wave with Trend":
            history_vals = (80 + 15 * np.sin(time / 10) + 0.15 * time + noise).tolist()
        elif signal_preset == "Upward Linear Drift":
            history_vals = (50 + 0.8 * time + noise).tolist()
        else:
            # Multi-seasonal demand cycle
            history_vals = (120 + 20 * np.sin(time / 6) + 10 * np.cos(time / 24) + noise).tolist()
            
        if btn_forecast:
            with st.spinner("Executing sequence forward pass..."):
                model_key = model_choice.lower().replace("-", "")
                
                try:
                    payload = {"history": history_vals, "model_type": model_key}
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    forecast_vals = r.json()["forecast"]
                except Exception:
                    forecast_vals = mock_predict(history_vals, model_key)
                    
                # Create interactive line chart
                fig = go.Figure()
                # History line
                fig.add_trace(go.Scatter(
                    x=list(range(96)), y=history_vals,
                    mode='lines', name='History (96 Steps)',
                    line=dict(color='#4FACFE', width=2)
                ))
                # Forecast line
                fig.add_trace(go.Scatter(
                    x=list(range(96, 120)), y=forecast_vals,
                    mode='lines+markers', name=f'Forecast ({model_choice} - 24 Steps)',
                    line=dict(color='#00F2FE', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Sequence Forecasting Results using {model_choice}",
                    xaxis_title="Time steps",
                    yaxis_title="Values",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#171B26",
                    font_color="#FFFFFF"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Draw preview plot
            fig_pre = go.Figure()
            fig_pre.add_trace(go.Scatter(x=list(range(96)), y=history_vals, mode='lines', name='Input History Preview', line=dict(color='#4FACFE')))
            fig_pre.update_layout(title="Input Sequence Preview (96 historical steps)", paper_bgcolor="#0E1117", plot_bgcolor="#171B26", font_color="#FFFFFF")
            st.plotly_chart(fig_pre, use_container_width=True)

# TAB 2: BACKTESTING & MODEL BENCHMARKS
with tab2:
    st.markdown("### Comparative Performance & Backtesting")
    
    # Load backtest data for all three models
    with st.spinner("Fetching model backtest runs..."):
        model_names = ["PatchTST", "N-BEATS", "TFT"]
        backtests = {}
        for name in model_names:
            m_key = name.lower().replace("-", "")
            try:
                r = requests.get(f"{API_URL}/backtest", params={"model_type": m_key}, timeout=15)
                backtests[name] = r.json()
            except Exception:
                backtests[name] = mock_backtest(m_key)
                
    # Metric Summary columns
    col_met1, col_met2, col_met3 = st.columns(3)
    
    with col_met1:
        st.markdown(f"""
        <div class='section-card'>
            <div class='metric-label'>PatchTST Average MAE</div>
            <div class='metric-value'>{backtests["PatchTST"]["metrics"]["MAE"]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_met2:
        st.markdown(f"""
        <div class='section-card'>
            <div class='metric-label'>N-BEATS Average MAE</div>
            <div class='metric-value'>{backtests["N-BEATS"]["metrics"]["MAE"]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_met3:
        st.markdown(f"""
        <div class='section-card'>
            <div class='metric-label'>Temporal Fusion Transformer MAE</div>
            <div class='metric-value'>{backtests["TFT"]["metrics"]["MAE"]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Main Comparison Chart
    st.markdown("#### Backtest Actuals vs Model Predictions")
    hist_len = len(backtests["PatchTST"]["history"])
    fore_len = len(backtests["PatchTST"]["actuals"])
    
    fig_comp = go.Figure()
    # History baseline
    fig_comp.add_trace(go.Scatter(
        x=list(range(hist_len)), 
        y=backtests["PatchTST"]["history"], 
        mode='lines', name='Historical History', line=dict(color='#8A8F98')
    ))
    # Actual future line
    fig_comp.add_trace(go.Scatter(
        x=list(range(hist_len, hist_len + fore_len)), 
        y=backtests["PatchTST"]["actuals"], 
        mode='lines', name='Actual Target', line=dict(color='#FFFFFF', width=3)
    ))
    
    # Model forecast overlays
    colors = {"PatchTST": "#00F2FE", "N-BEATS": "#FF007F", "TFT": "#FFD700"}
    for name in model_names:
        fig_comp.add_trace(go.Scatter(
            x=list(range(hist_len, hist_len + fore_len)), 
            y=backtests[name]["forecast"], 
            mode='lines+markers', name=f'{name} Prediction', line=dict(color=colors[name], width=2)
        ))
        
    fig_comp.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#171B26", font_color="#FFFFFF")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Error bar charts
    st.markdown("#### Forecast Metrics Comparison (Lower is Better)")
    metric_rows = []
    for name in model_names:
        metrics = backtests[name]["metrics"]
        metric_rows.append({"Model": name, "Metric": "MAE", "Error": metrics["MAE"]})
        metric_rows.append({"Model": name, "Metric": "RMSE", "Error": metrics["RMSE"]})
        
    metrics_df = pd.DataFrame(metric_rows)
    fig_bar = px.bar(metrics_df, x="Model", y="Error", color="Metric", barmode="group", color_discrete_map={"MAE": "#4FACFE", "RMSE": "#FF007F"})
    fig_bar.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#171B26", font_color="#FFFFFF")
    st.plotly_chart(fig_bar, use_container_width=True)
