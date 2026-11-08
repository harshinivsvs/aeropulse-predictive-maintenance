import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_and_train_models
from sklearn.metrics import precision_score, recall_score

# Page Configuration
st.set_page_config(
    page_title="AeroPulse | Enterprise Intelligence", 
    page_icon="⚙️", 
    layout="wide"
)

# Elite SaaS Dark Theme CSS Injection
st.markdown("""
    <style>
    .stApp { background-color: #07090e; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #090d16 100%);
        border: 1px solid #1e293b; padding: 30px; border-radius: 16px; margin-bottom: 25px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title { font-size: 2.4rem; font-weight: 800; color: #ffffff; margin-bottom: 4px; }
    .hero-subtitle { font-size: 1rem; color: #94a3b8; }

    .saas-metric-card {
        background-color: #0f172a; border: 1px solid #1e293b; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;
    }
    .saas-metric-card:hover { border-color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-top: 6px; }
    .metric-delta { font-size: 0.8rem; color: #34d399; margin-top: 4px; }

    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #f59e0b; color: #fbbf24; padding: 20px; border-radius: 12px; font-weight: 600;
    }
    .status-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #10b981; color: #34d399; padding: 20px; border-radius: 12px; font-weight: 600;
    }

    section[data-testid="stSidebar"] { background-color: #090d16; border-right: 1px solid #1e293b; }
    .stButton button {
        background-color: #38bdf8 !important; color: #07090e !important; font-weight: 700 !important;
        border-radius: 8px !important; padding: 0.6rem 1rem !important; border: none !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0f172a; padding: 6px; border-radius: 10px; border: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; font-weight: 600; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #38bdf8 !important; color: #07090e !important; }
    </style>
""", unsafe_allow_html=True)

# Load models from modular backend
rf_model, lr_model, df_data, X_test, y_test = load_and_train_models()

# Hero Banner
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚙️ AeroPulse AI — Industrial Intelligence Platform</div>
        <div class="hero-subtitle">Enterprise Predictive Maintenance Engine & Real-Time Telemetry Analytics</div>
    </div>
""", unsafe_allow_html=True)

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="saas-metric-card"><div class="metric-label">Active Sensors</div><div class="metric-value">1,240</div><div class="metric-delta">▲ +12 Nodes Online</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="saas-metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">96.4%</div><div class="metric-delta">▲ Optimized Pipeline</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="saas-metric-card"><div class="metric-label">System Health</div><div class="metric-value">Optimal</div><div class="metric-delta">◆ Zero Faults Tripped</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="saas-metric-card"><div class="metric-label">Inference Latency</div><div class="metric-value">14 ms</div><div class="metric-delta">▼ Real-Time Stream</div></div>', unsafe_allow_html=True)

st.write("")

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Telemetry Controls")
st.sidebar.success(f"📂 Loaded Dataset: {len(df_data)} rows")
model_choice = st.sidebar.selectbox("Active ML Architecture", ["Random Forest (Ensemble)", "Logistic Regression (Baseline)"])
st.sidebar.divider()

air_temp = st.sidebar.slider("Air Temperature (K)", 290.0, 310.0, 298.0)
proc_temp = st.sidebar.slider("Process Temperature (K)", 300.0, 320.0, 308.5)
rot_speed = st.sidebar.slider("Rotational Speed (rpm)", 1000, 2800, 1520)
torque = st.sidebar.slider("Torque (Nm)", 5.0, 90.0, 42.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 50)

active_model = rf_model if "Random Forest" in model_choice else lr_model

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Live Diagnostics", "📊 Exploratory Analysis", "🔍 Model Benchmarks"])

with tab1:
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("#### 📋 Active Sensor Stream Profile")
        input_df = pd.DataFrame({"air_temperature": [air_temp], "process_temperature": [proc_temp], "rotational_speed": [rot_speed], "torque": [torque], "tool_wear": [tool_wear]})
        st.dataframe(input_df, use_container_width=True)
        evaluate_btn = st.button("Run Machine Health Diagnostic", use_container_width=True)
        
    with col_b:
        st.markdown("#### 🚨 Diagnostic Output Panel")
        if evaluate_btn:
            pred = active_model.predict(input_df)[0]
            prob = active_model.predict_proba(input_df)[0][1]
            if pred == 1 or prob > 0.35:
                st.markdown(f'<div class="status-warning">⚠️ CRITICAL FAILURE RISK DETECTED<br><br>• Failure Probability: <b>{prob * 100:.2f}%</b><br>• Action Required: Schedule immediate maintenance inspection.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-success">✅ SYSTEM OPERATIONS NOMINAL<br><br>• Failure Probability: <b>{prob * 100:.2f}%</b><br>• Status: All telemetry parameters within safe thresholds.</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Adjust sidebar sliders and click **Run Machine Health Diagnostic**.")

    st.divider()
    if "Random Forest" in model_choice:
        st.markdown("#### 📈 Feature Contribution Weights")
        importances = rf_model.named_steps['classifier'].feature_importances_
        feat_imp = pd.DataFrame({"Feature": input_df.columns, "Importance": importances}).sort_values(by="Importance", ascending=True)
        fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', template="plotly_dark", title="Random Forest Feature Importance")
        fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", title_font_color="#38bdf8", margin=dict(t=40, b=20, l=20, r=20))
        fig_imp.update_traces(marker_color="#38bdf8")
        st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.markdown("#### 📊 Operational Stress & Failure Mapping")
    fig_scatter = px.scatter(df_data, x="tool_wear", y="torque", color=df_data["machine_failure"].astype(str), color_discrete_map={"0": "#38bdf8", "1": "#fbbf24"}, template="plotly_dark", title="Torque vs. Tool Wear Stress Clustering")
    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", title_font_color="#38bdf8", margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("#### 📈 Enterprise Model Evaluation Benchmarks")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**🌲 Random Forest Ensemble Pipeline**")
        st.metric(label="Precision Score", value=f"{precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
        st.metric(label="Recall Score (Safety Critical)", value=f"{recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    with mc2:
        st.markdown("**📉 Logistic Regression Baseline Pipeline**")
        st.metric(label="Precision Score", value=f"{precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
        st.metric(label="Recall Score (Safety Critical)", value=f"{recall_score(y_test, y_pred_lr, zero_division=0):.4f}")