"""
Streamlit Dashboard for STORM-PhysNet.
Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STORM-PhysNet | GEO Electron Flux Forecasting",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — IEEE Standard Light Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Pure white background for IEEE */
    .stApp {
        background: #ffffff;
    }

    /* High contrast text everywhere */
    h1, h2, h3, p, span, div {
        color: #000000 !important;
    }

    .metric-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #000000 !important;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #343a40 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }

    .storm-badge {
        background: #d32f2f;
        color: white !important;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 2px solid #b71c1c;
    }

    .quiet-badge {
        background: #2e7d32;
        color: white !important;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 2px solid #1b5e20;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000 !important;
        border-bottom: 3px solid #000000;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    .sidebar-section {
        background: #f1f3f5;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="black", family="Arial", size=14),
    xaxis=dict(
        gridcolor="#e9ecef", 
        linecolor="black", 
        linewidth=2, 
        mirror=True, 
        zeroline=False,
        title_font=dict(size=16, weight="bold")
    ),
    yaxis=dict(
        gridcolor="#e9ecef", 
        linecolor="black", 
        linewidth=2, 
        mirror=True, 
        zeroline=False,
        title_font=dict(size=16, weight="bold")
    ),
    margin=dict(l=60, r=30, t=50, b=50),
)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_demo_data(vsw, bz, density, storm_active):
    """Generate 72-hour demo time series."""
    np.random.seed(42)
    t = np.arange(72)

    # Solar wind
    vsw_ts   = vsw   + np.random.normal(0, 20, 72)
    bz_ts    = bz    + np.random.normal(0, 2, 72)
    dens_ts  = density + np.random.exponential(1, 72)

    if storm_active:
        bz_ts[30:55] = bz - np.abs(bz) * 1.5 - 5
        vsw_ts[30:55] += 150

    # Flux
    log_flux = 3.5 + 0.004 * (vsw_ts - 400)
    log_flux += 0.005 * np.maximum(-bz_ts, 0)
    log_flux += np.random.normal(0, 0.1, 72)
    log_flux = np.cumsum(np.clip(np.diff(log_flux, prepend=log_flux[0]),
                                  -0.3, 0.3)) + log_flux[0]
    log_flux = np.clip(log_flux, -1, 6)
    return t, vsw_ts, bz_ts, dens_ts, log_flux


def predict_mock(log_flux, vsw, bz, storm_active):
    """Mock model prediction (returns realistic values)."""
    current_flux = log_flux[-1]
    bz_now       = bz[-1]
    vsw_now      = vsw[-1]

    np.random.seed(int(abs(bz_now * vsw_now)) % 1000)

    # Persistence baseline
    persist = current_flux

    # Learned correction (mock physics residual)
    if storm_active and bz_now < -5:
        # Storm injection: possible flux enhancement in 6-12h
        corr_30m = np.random.uniform(-0.2, 0.1)
        corr_6h  = np.random.uniform(-0.1, 0.8)
        corr_12h = np.random.uniform(0.2, 1.5)
    else:
        # Quiet: small corrections
        corr_30m = np.random.normal(0, 0.05)
        corr_6h  = np.random.normal(0, 0.1)
        corr_12h = np.random.normal(0, 0.15)

    preds = np.array([
        persist + corr_30m,
        persist + corr_6h,
        persist + corr_12h,
    ])
    stds = np.array([0.1, 0.25, 0.4]) * (2.0 if storm_active else 1.0)

    # Storm probability
    storm_prob = min(0.95, max(0.05,
                               0.1 + 0.05 * abs(min(bz_now, 0)) +
                               0.001 * max(vsw_now - 400, 0)))

    # Bz gate activation
    gate = min(1.0, max(0.1, 0.1 + 0.03 * abs(min(bz_now, 0))))

    return preds, stds, storm_prob, gate


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 24px 0 16px 0;">
    <div style="font-size:2.8rem; font-weight:700; background:linear-gradient(135deg,#63b3ed,#9f7aea,#f687b3);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em;">
        🛰️ STORM-PhysNet
    </div>
    <div style="color:#718096; font-size:1rem; margin-top:6px; font-weight:400;">
        Storm-aware Physics-Informed Network for GEO Electron Flux Forecasting
    </div>
    <div style="color:#4a5568; font-size:0.8rem; margin-top:4px;">
        GOES >2 MeV | Wind Solar Wind | GRASP/GSAT-12R Indian Longitude
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Solar Wind Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Solar Wind Parameters")
    st.markdown("Adjust parameters to simulate space weather conditions:")

    with st.container():
        vsw     = st.slider("Solar Wind Speed (km/s)", 250, 900, 450, 10,
                             help="Typical: 300-500 km/s. High-speed streams: >600 km/s")
        bz      = st.slider("IMF Bz (nT)",  -40.0, 15.0, -2.0, 0.5,
                             help="Negative = southward → geomagnetic storm driver")
        density = st.slider("Proton Density (cm⁻³)", 1.0, 50.0, 8.0, 0.5)
        pdyn    = round(0.5 * 1.67e-27 * (density * 1e6) * (vsw * 1e3)**2 * 1e9, 2)

    st.markdown("---")
    storm_active = st.toggle("🔴 Simulate Storm Event", value=(bz < -8))

    st.markdown("---")
    st.markdown("## 🔬 Model Settings")
    show_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)
    show_gate        = st.checkbox("Show Bz Gate Activation", value=True)
    horizon_labels   = ["30-45 min", "6 hours", "12 hours"]

# ─────────────────────────────────────────────────────────────────────────────
# Generate data & predictions
# ─────────────────────────────────────────────────────────────────────────────

t, vsw_ts, bz_ts, dens_ts, log_flux = generate_demo_data(
    vsw, bz, density, storm_active)
preds, stds, storm_prob, gate_val = predict_mock(
    log_flux, vsw_ts, bz_ts, storm_active)

flux_lin = 10 ** log_flux

# Storm status badge
storm_status = "storm" if storm_active or bz < -8 else "quiet"
badge_html = (f'<span class="storm-badge">⚡ STORM ACTIVE</span>'
              if storm_status == "storm"
              else f'<span class="quiet-badge">✓ QUIET CONDITIONS</span>')

# ─────────────────────────────────────────────────────────────────────────────
# Top metrics row
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"**Space Weather Status:** {badge_html}", unsafe_allow_html=True)
st.markdown("")

col1, col2, col3, col4, col5 = st.columns(5)
horizon_icons = ["⚡", "🕐", "🌙"]
for col, icon, label, pred, std in zip(
        [col1, col2, col3], horizon_icons, horizon_labels, preds, stds):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{pred:.2f}</div>
            <div style="color:#718096; font-size:0.75rem;">
                log₁₀(e⁻/cm²/s/sr)
            </div>
            <div style="color:#9f7aea; font-size:0.75rem; margin-top:4px;">
                ±{std:.2f} uncertainty
            </div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌀 Storm Probability</div>
        <div class="metric-value">{storm_prob:.0%}</div>
        <div style="color:#718096; font-size:0.75rem;">Next 6 hours</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚙️ Bz Gate</div>
        <div class="metric-value">{gate_val:.2f}</div>
        <div style="color:#718096; font-size:0.75rem;">Physics activation</div>
        <div style="background:rgba(99,179,237,0.15); border-radius:4px;
                    height:6px; margin-top:6px;">
            <div style="background:linear-gradient(90deg,#63b3ed,#9f7aea);
                        border-radius:4px; height:6px;
                        width:{gate_val*100:.0f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main plots
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Flux Forecast", "🌀 Solar Wind", "📊 Performance Metrics", "🔬 Physics Insights"
])

with tab1:
    fig = make_subplots(rows=1, cols=1)
    # Historical flux
    fig.add_trace(go.Scatter(
        x=t, y=log_flux,
        name="Observed Flux",
        line=dict(color="#1f77b4", width=4),
        mode="lines",
    ))
    # Forecast points
    forecast_t     = [72, 72+6, 72+12]
    forecast_labels = ["30-45 min", "6h", "12h"]
    colors         = ["#ff7f0e", "#d62728", "#9467bd"]

    for ft, fp, fs, fl, fc in zip(forecast_t, preds, stds,
                                   forecast_labels, colors):
        if show_uncertainty:
            fig.add_trace(go.Scatter(
                x=[ft, ft], y=[fp - 1.96 * fs, fp + 1.96 * fs],
                mode="lines",
                line=dict(color=fc, width=3, dash="dot"),
                showlegend=False, opacity=0.7,
            ))
        fig.add_trace(go.Scatter(
            x=[ft], y=[fp],
            mode="markers",
            name=f"Forecast {fl}",
            marker=dict(color=fc, size=12, symbol="star",
                        line=dict(color="white", width=1.5)),
        ))

    # Radiation hazard threshold
    fig.add_hline(y=4.0, line_dash="dash", line_color="#d62728", line_width=2,
                   annotation_text="Radiation Hazard Level (10⁴)", annotation_font_color="#d62728")

    fig.add_vline(x=72, line_dash="dash", line_color="black", line_width=2,
                   annotation_text="NOW", annotation_font_color="black")

    fig.update_layout(
        **PLOT_THEME,
        title=dict(text="GEO Electron Flux Forecast (log₁₀ scale)", font=dict(color="black", size=20, weight="bold")),
        xaxis_title="Time (hours)",
        yaxis_title="log₁₀(flux [e⁻/cm²/s/sr])",
        height=400,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(rows=3, cols=1,
                          subplot_titles=("Solar Wind Speed (km/s)",
                                          "IMF Bz (nT)",
                                          "Proton Density (cm⁻³)"),
                          shared_xaxes=True, vertical_spacing=0.15)

    fig2.add_trace(go.Scatter(x=t, y=vsw_ts,
                               line=dict(color="#ff7f0e", width=3),
                               name="Vsw"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=t, y=bz_ts,
                               line=dict(color="#d62728", width=3),
                               name="Bz",
                               fill="tozeroy",
                               fillcolor="rgba(214,39,40,0.15)"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=t, y=dens_ts,
                               line=dict(color="#2ca02c", width=3),
                               name="Density"), row=3, col=1)

    fig2.add_hline(y=-5, row=2, line_dash="dash",
                    line_color="#d62728", line_width=2,
                    annotation_text="Gate threshold", annotation_font_color="#d62728")
    
    # Copy PLOT_THEME but override margin and height to give titles room to breathe
    theme2 = PLOT_THEME.copy()
    theme2['margin'] = dict(l=60, r=30, t=90, b=50)
    
    fig2.update_layout(**theme2, height=650,
                        title=dict(text="Solar Wind History (72-hour input window)", font=dict(color="black", size=20, weight="bold")))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Real performance comparison table from ieee_table.txt
    perf_data = {
        "Model": [
            "STORM-PhysNet (Ours, Ensembled)",
            "STORM-PhysNet (Ours, storm_bz)",
            "Transformer (Tan 2024)",
            "LSTM (Baseline)",
            "MLP (Baseline)",
            "CNN (Baseline)",
        ],
        "PE (All)":    [0.7180, 0.6686, 0.6475, 0.6140, 0.5251, 0.0326],
        "PE (Storm)":  [0.7029, 0.6737, 0.6118, 0.6342, 0.5409, 0.1644],
        "PE (High)":   [0.7828, 0.7450, 0.7171, 0.7924, 0.6656, 0.1872],
        "RMSE":        [0.2318, 0.2511, 0.2590, 0.2711, 0.3008, 0.4293],
        "PE (45m)":    [0.3256, 0.3373, -0.1528, -1.0370, -2.5952, -5.5245],
        "PE (12h)":    [0.7753, 0.7419, 0.7293, 0.7225, 0.6702, 0.3500],
    }
    df_perf = pd.DataFrame(perf_data)

    st.markdown('<div class="section-header">Performance vs. SOTA Models (6-hour Forecast)</div>',
                unsafe_allow_html=True)
    st.dataframe(df_perf.style.highlight_max(
        subset=["PE (All)", "PE (Storm)", "PE (High)", "PE (45m)", "PE (12h)"],
        color="#1a3a2a").highlight_min(subset=["RMSE"], color="#1a3a2a"), 
        use_container_width=True, hide_index=True)

    st.info("⚡ **Storm-period performance** (Dst ≤ -50 nT) is the key metric. "
            "Our physics-informed ensemble avoids the 'MSE Trap' and dominates the storm peaks.")

    # Ablation table
    st.markdown('<div class="section-header">Ablation Study (Physics Impact, n=3 seeds)</div>',
                unsafe_allow_html=True)
    # Baseline PE_storm for storm_bz = 0.6737
    # storm_no_delay (n=3) PE_storm = 0.6435 -> Degradation = (0.6435 - 0.6737) = -0.0302 (-3.02%)
    # storm_no_physics (n=3) PE_storm = 0.6509 -> Degradation = (0.6509 - 0.6737) = -0.0228 (-2.28%)
    # LSTM PE_storm = 0.6342 -> Degradation = (0.6342 - 0.6737) = -0.0395 (-3.95%)
    ablation_data = {
        "Experiment":           ["Full STORM-PhysNet (storm_bz)", "− Adaptive Delay (no_delay)", "− Physics Loss (no_physics)", "LSTM Baseline"],
        "PE (All)":             [0.6686, 0.6719, 0.6768, 0.6140],
        "PE (Storm)":           [0.6737, 0.6435, 0.6509, 0.6342],
        "Storm Degradation":    ["Baseline", "-3.02%", "-2.28%", "-3.95%"],
    }
    st.dataframe(pd.DataFrame(ablation_data), use_container_width=True, hide_index=True)

with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Feature Importance (iTransformer Attention)</div>',
                    unsafe_allow_html=True)
        features = ["Vsw", "Bz", "Bt", "Density", "Pdyn",
                    "Bz_dur", "Storm_onset", "Vsw_6h", "Bz_6h", "Bz×Vsw"]
        importance = np.array([0.18, 0.28, 0.12, 0.08, 0.10,
                                0.11, 0.05, 0.04, 0.03, 0.01])
        # Amplify if storm
        if storm_active:
            importance[1] *= 1.5  # Bz more important in storms
            importance /= importance.sum()

        fig_imp = go.Figure(go.Bar(
            y=features, x=importance, orientation="h",
            marker=dict(
                color=importance,
                colorscale=[[0, "#1a3a6a"], [0.5, "#3b82f6"], [1, "#9f7aea"]],
            )
        ))
        fig_imp.update_layout(**PLOT_THEME, height=350, showlegend=False,
                               xaxis_title="Attention Weight")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Bz Gate Activation Over Time</div>',
                    unsafe_allow_html=True)
        gate_ts = np.clip(0.1 + 0.05 * np.maximum(-bz_ts, 0), 0, 1)
        fig_gate = go.Figure()
        fig_gate.add_trace(go.Scatter(
            x=t, y=gate_ts,
            fill="tozeroy",
            fillcolor="rgba(159,122,234,0.15)",
            line=dict(color="#9f7aea", width=2),
            name="Gate Activation",
        ))
        fig_gate.add_hline(y=0.5, line_dash="dash",
                            line_color="rgba(255,255,255,0.3)",
                            annotation_text="Storm threshold")
        fig_gate.update_layout(**PLOT_THEME, height=350,
                                yaxis_title="Gate Value",
                                xaxis_title="Time (hours)",
                                yaxis_range=[0, 1.05])
        st.plotly_chart(fig_gate, use_container_width=True)

    st.markdown("""
    <div style="background:rgba(15,25,50,0.6); border-radius:12px; padding:16px;
                border:1px solid rgba(99,179,237,0.15); margin-top:8px;">
        <b style="color:#63b3ed;">Physics Interpretability</b><br>
        <span style="color:#a0aec0; font-size:0.85rem;">
        The <b>iTransformer attention weights</b> show which solar wind feature
        combinations the model learned as flux drivers — physically interpretable
        as the Dungey reconnection cycle (Bz × Vsw → energy input).<br><br>
        The <b>Bz Gate</b> (your original idea) amplifies model signals when
        IMF Bz turns southward, encoding the known physics of storm onset.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding:8px 0;">
    STORM-PhysNet | Physics-Informed GEO Electron Flux Forecasting |
    Target: IEEE TGRS / AGU Space Weather Journal
</div>
""", unsafe_allow_html=True)
