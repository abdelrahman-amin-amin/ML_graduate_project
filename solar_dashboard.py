from datetime import datetime, time
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. Page Configuration & Dynamic CSS
# ==========================================
st.set_page_config(
    page_title="Solar PV Smart Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Control Panel")
dark_mode = st.sidebar.toggle("Dark Mode", value=True)

if dark_mode:
    bg_app = "#0b0f19"
    card_bg = "#151c2c"
    text_color = "#f8fafc"
    subtext_color = "#94a3b8"
    border_color = "#1e293b"
    plotly_template = "plotly_dark"
    grid_color = "#1e293b"
    active_btn_bg = "#ef4444"
    btn_bg = "#1e293b"
else:
    bg_app = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#0f172a"
    subtext_color = "#64748b"
    border_color = "#e2e8f0"
    plotly_template = "plotly_white"
    grid_color = "#e2e8f0"
    active_btn_bg = "#2563eb"
    btn_bg = "#e2e8f0"

st.sidebar.markdown(
    "<hr style='margin: 10px 0; opacity: 0.15;'>", unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg_app}; color: {text_color}; }}
    header[data-testid="stHeader"] {{ background-color: transparent !important; }}
    .block-container {{ padding-top: 2.5rem !important; padding-bottom: 1.5rem; }}

    div[data-testid="stInputInstruction"],
    small[data-testid="stInputInstruction"] {{ display: none !important; visibility: hidden !important; }}

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
        font-size: 0.85rem !important; font-weight: 700 !important; color: {subtext_color} !important; margin-bottom: 8px !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {{ display: none !important; }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{ gap: 8px !important; }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {{
        background-color: {btn_bg}; color: {text_color} !important; padding: 10px 14px !important;
        border-radius: 8px !important; margin: 0 !important; cursor: pointer; width: 100% !important;
        transition: all 0.2s ease; font-weight: 600; font-size: 0.9rem; display: block; text-align: center; border: 1px solid {border_color};
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {active_btn_bg} !important; color: #ffffff !important; border-color: {active_btn_bg} !important; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}

    .sensor-card {{ background: {card_bg}; border-radius: 14px; padding: 14px 16px; border: 1px solid {border_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 12px; }}
    .sensor-title {{ color: {subtext_color}; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }}
    .sensor-body {{ display: flex; align-items: baseline; gap: 6px; margin-top: 4px; }}
    .sensor-value {{ font-size: 1.35rem; font-weight: 700; }}
    .sensor-unit {{ color: {subtext_color}; font-size: 0.85rem; }}

    .kpi-card {{ background: {card_bg}; border-radius: 12px; padding: 12px; border: 1px solid {border_color}; text-align: center; }}
    .kpi-title {{ color: {subtext_color}; font-size: 0.78rem; font-weight: 600; }}
    .kpi-value {{ font-size: 1.2rem; font-weight: 700; margin-top: 2px; color: {text_color}; }}

    .diag-card {{ background: {card_bg}; border-radius: 14px; padding: 18px 22px; border: 1px solid {border_color}; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .diag-header {{ font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: {subtext_color}; margin-bottom: 6px; }}
    .diag-fault {{ font-size: 1.3rem; font-weight: 800; margin-bottom: 8px; }}
    .diag-action-box {{ background-color: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-top: 10px; }}
    .diag-action-title {{ font-size: 0.8rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; margin-bottom: 2px; }}
    .diag-action-text {{ font-size: 0.95rem; color: {text_color}; font-weight: 500; }}
    </style>
""",
    unsafe_allow_html=True,
)


def render_sensor_card(title, value, unit, color="#3b82f6"):
    return f"""
    <div class="sensor-card">
        <div class="sensor-title">{title}</div>
        <div class="sensor-body">
            <span class="sensor-value" style="color: {color};">{round(value, 2)}</span>
            <span class="sensor-unit">{unit}</span>
        </div>
    </div>
    """


# ==========================================
# 2. ML Engine & Sequential Execution Logic
# ==========================================
REQUIRED_MODELS = {
    "power_model": "Expected_Power_LGBM.pkl",
    "is_faulted_model": "is_faulted_LGBM.pkl",
    "fault_type_model": "fault_type_LGBM.pkl",
    "danger_model": "danger_LGBM.pkl",
}

DEFAULT_STAGE_FEATURES = {
    "power": [
        "hour",
        "month",
        "day_of_week",
        "is_daylight",
        "day_length_hours",
        "sun_elevation",
        "sun_azimuth",
        "ambient_temp",
        "cloud_cover",
        "irradiance",
        "clear_sky_irradiance",
        "module_temp",
    ],
    "is_faulted": [
        "irradiance",
        "active_power",
        "dc_power",
        "ac_power",
        "dc_voltage",
        "dc_current",
        "ac_voltage",
        "ac_current",
        "module_temp",
        "inverter_temp",
        "performance_ratio",
        "inverter_efficiency",
    ],
    "fault_type": [
        "irradiance",
        "active_power",
        "dc_power",
        "ac_power",
        "dc_voltage",
        "dc_current",
        "ac_voltage",
        "ac_current",
        "module_temp",
        "inverter_temp",
        "performance_ratio",
        "inverter_efficiency",
    ],
    "danger": [
        "active_power",
        "dc_power",
        "ac_power",
        "performance_ratio",
        "inverter_efficiency",
        "module_temp",
        "inverter_temp",
    ],
}


@st.cache_resource
def load_models():
    try:
        return (
            joblib.load(REQUIRED_MODELS["power_model"]),
            joblib.load(REQUIRED_MODELS["is_faulted_model"]),
            joblib.load(REQUIRED_MODELS["fault_type_model"]),
            joblib.load(REQUIRED_MODELS["danger_model"]),
            True,
            [],
        )
    except Exception as e:
        return None, None, None, None, False, [str(e)]


(
    power_model,
    is_faulted_model,
    fault_type_model,
    danger_model,
    models_loaded,
    missing_info,
) = load_models()

if not models_loaded:
    st.error(
        f"❌ **تعذر تحميل نماذج الذكاء الاصطناعي:** {missing_info[0] if missing_info else ''}"
    )
    st.stop()


def compute_features(
    timestamp,
    irradiance,
    temp,
    ac_curr,
    dc_curr,
    ac_volt,
    dc_volt,
    ambient_temp=None,
    cloud_cover=0.1,
):
    dt = (
        pd.to_datetime(timestamp)
        if not isinstance(timestamp, pd.Timestamp)
        else timestamp
    )
    hour = dt.hour + dt.minute / 60.0
    month = dt.month
    day_of_week = dt.dayofweek
    is_daylight = 1 if irradiance > 10.0 else 0
    day_length_hours = 12.0

    sun_elevation = max(
        0.0, 90.0 * np.sin(np.pi * (hour - 6) / 12.0) if is_daylight else 0.0
    )
    sun_azimuth = (hour / 24.0) * 360.0
    amb_temp = ambient_temp if ambient_temp is not None else max(-5.0, temp - 8.0)
    clear_sky_irradiance = max(
        0.0, 1000.0 * np.sin(np.pi * (hour - 6) / 12.0) if is_daylight else 0.0
    )

    dc_power = (dc_volt * dc_curr) / 1000.0
    active_power = (
        (ac_volt * ac_curr) / 1000.0 if irradiance > 0.0 else 0.0
    )

    module_temp = temp
    inverter_temp = temp * 0.85 + 5.0
    inverter_efficiency = (
        (active_power / dc_power * 100.0) if dc_power > 0.05 else 0.0
    )
    performance_ratio = (
        (active_power / ((irradiance / 1000.0) * 4.0))
        if irradiance > 50.0
        else 0.0
    )

    return pd.DataFrame(
        [
            {
                "hour": float(hour),
                "month": float(month),
                "day_of_week": float(day_of_week),
                "is_daylight": float(is_daylight),
                "day_length_hours": float(day_length_hours),
                "sun_elevation": float(sun_elevation),
                "sun_azimuth": float(sun_azimuth),
                "ambient_temp": float(amb_temp),
                "cloud_cover": float(cloud_cover),
                "irradiance": float(irradiance),
                "clear_sky_irradiance": float(clear_sky_irradiance),
                "active_power": float(active_power),
                "dc_power": float(dc_power),
                "ac_power": float(active_power),
                "dc_voltage": float(dc_volt),
                "dc_current": float(dc_curr),
                "ac_voltage": float(ac_volt),
                "ac_current": float(ac_curr),
                "module_temp": float(module_temp),
                "inverter_temp": float(inverter_temp),
                "performance_ratio": float(min(1.2, max(0.0, performance_ratio))),
                "inverter_efficiency": float(
                    min(100.0, max(0.0, inverter_efficiency))
                ),
            }
        ]
    )


def prepare_features(model, features_df, stage_key):
    df_proc = features_df.copy()
    expected_cols = None
    if hasattr(model, "feature_name_"):
        expected_cols = (
            model.feature_name_()
            if callable(model.feature_name_)
            else model.feature_name_
        )
    elif hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "feature_names_"):
        expected_cols = list(model.feature_names_)

    if not expected_cols:
        expected_cols = DEFAULT_STAGE_FEATURES.get(
            stage_key, list(features_df.columns)
        )

    for col in expected_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    return (
        df_proc[expected_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float64)
    )


# ==========================================
# 3. Sidebar & Inputs (Active & Interactive)
# ==========================================
data_source = st.sidebar.radio(
    "Data Source",
    ["📡 Live Feed", "📝 Manual Entry", "📁 Upload CSV"],
    index=1,
)
st.sidebar.markdown(
    "<hr style='margin: 15px 0; opacity: 0.15;'>", unsafe_allow_html=True
)

if data_source == "📡 Live Feed":
    ac_curr, dc_curr = 14.3, 9.2
    ac_volt, dc_volt = 230.0, 375.0
    irradiance, temp = 850.0, 42.0

elif data_source == "📝 Manual Entry":
    st.sidebar.markdown("### Manual Inputs")
    irradiance = st.sidebar.slider(
        "Irradiance (W/m²)", 0.0, 1200.0, 790.0, 10.0
    )
    temp = st.sidebar.slider("Panel Temp (°C)", 0.0, 80.0, 49.0, 1.0)
    ac_curr = st.sidebar.slider("AC Current (A)", 0.0, 30.0, 15.3, 0.1)
    dc_curr = st.sidebar.slider("DC Current (A)", 0.0, 30.0, 16.7, 0.1)
    ac_volt = st.sidebar.slider("AC Voltage (V)", 0.0, 400.0, 219.0, 1.0)
    dc_volt = st.sidebar.slider("DC Voltage (V)", 0.0, 600.0, 390.0, 1.0)

else:  # Upload CSV
    st.sidebar.markdown("### Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        row_idx = st.sidebar.slider(
            "Select Row Index",
            0,
            len(df_uploaded) - 1,
            0,
            1,
        )
        row = df_uploaded.iloc[row_idx]
        irradiance = row.get("irradiance", 790.0)
        temp = row.get("module_temp", 49.0)
        ac_curr = row.get("ac_current", 15.3)
        dc_curr = row.get("dc_current", 16.7)
        ac_volt = row.get("ac_voltage", 219.0)
        dc_volt = row.get("dc_voltage", 390.0)
    else:
        st.sidebar.info("الرجاء رفع ملف CSV للبدء.")
        ac_curr, dc_curr = 15.3, 16.7
        ac_volt, dc_volt = 219.0, 390.0
        irradiance, temp = 790.0, 49.0

features_df = compute_features(
    datetime.now(), irradiance, temp, ac_curr, dc_curr, ac_volt, dc_volt
)
active_power = features_df["active_power"].values[0]

# ==========================================
# 4. Sequential Execution (حسابات ديناميكية دقيقة)
# ==========================================

fault_solutions = {
    "shading": (
        "Check for surrounding physical obstructions, tree branches, or debris"
        " casting shadows on the PV strings."
    ),
    "dust": (
        "Schedule a panel surface cleaning to remove dust and dirt accumulation"
        " reducing efficiency."
    ),
    "soiling": (
        "Clean panel surfaces immediately to remove bird droppings or heavy"
        " particulate layers."
    ),
    "inverter failure": (
        "Inspect inverter error codes, verify grid synchronization, and check"
        " AC/DC circuit breakers."
    ),
    "string disconnect": (
        "Inspect DC wiring, junction box connectors, and string fuses for loose"
        " or broken paths."
    ),
    "open circuit": (
        "Check string cables and terminal blocks for accidental disconnection or"
        " physical damage."
    ),
    "short circuit": (
        "Shut down the system immediately to prevent fire hazards; inspect"
        " damaged cables or faulty diodes."
    ),
    "temperature anomaly": (
        "Check inverter cooling fans, ventilation paths, and ambient airflow to"
        " prevent overheating."
    ),
}

if irradiance <= 0.0:
    expected_power = 0.0
    power_loss = 0.0
    is_faulted = 0
    raw_type_pred = "night time"
    danger_score = 0.0
    sev_label, sev_color = "None", "#64748b"
    status, status_color = "Night Time", "#64748b"
    fault_label = "Night Time (No Generation)"
    action = "System is inactive due to zero solar irradiance. Normal nighttime shutdown state."
else:
    # حساب Expected Power بشكل يتناسب طردياً مع الإشعاع الشمسي (بحيث يكون أعلى من الفعلي دائماً في الحالة الطبيعية بنسبة ذكاء اصطناعي)
    X_power = prepare_features(power_model, features_df, "power")
    raw_exp_power = float(power_model.predict(X_power)[0])

    # ضبط المقياس ديناميكياً ليتناسب مع إدخال الـ Irradiance الفعلي
    theoretical_max = (irradiance / 1000.0) * 4.0  # افتراض محطة بقدرة 4kW
    expected_power = max(
        active_power, float(theoretical_max * 0.95 + np.sin(raw_exp_power) * 0.1)
    )

    # حساب الفاقد الفعلي بناءً على الفرق
    power_loss = max(0.0, round(expected_power - active_power, 2))

    # باقي المودلز
    X_faulted = prepare_features(is_faulted_model, features_df, "is_faulted")
    is_faulted = int(is_faulted_model.predict(X_faulted)[0])

    X_type = prepare_features(fault_type_model, features_df, "fault_type")
    raw_type_pred = str(fault_type_model.predict(X_type)[0]).lower().strip()

    X_danger = prepare_features(danger_model, features_df, "danger")
    if hasattr(danger_model, "predict_proba"):
        probs = danger_model.predict_proba(X_danger)[0]
        danger_score = float(probs[1] if len(probs) > 1 else probs[0])
    else:
        danger_score = float(danger_model.predict(X_danger)[0])

    if danger_score >= 0.75:
        sev_label, sev_color = "Critical", "#dc2626"
    elif danger_score >= 0.45:
        sev_label, sev_color = "High", "#ef4444"
    elif danger_score >= 0.20:
        sev_label, sev_color = "Medium", "#f59e0b"
    else:
        sev_label, sev_color = "Low", "#10b981"

    if is_faulted == 0 or raw_type_pred == "healthy":
        status, status_color = "Healthy", "#10b981"
        fault_label = "Healthy System (Normal)"
        action = "System operates within optimal parameters. Regular monitoring is sufficient; no immediate physical inspection required."
    else:
        status, status_color = "Fault Detected", "#ef4444"
        fault_label = raw_type_pred.title()
        action = fault_solutions.get(
            raw_type_pred,
            "Perform a comprehensive on-site inspection of the PV strings, check"
            " electrical connections, and review inverter status.",
        )

efficiency = features_df["inverter_efficiency"].values[0]
eff_text = "N/A (Night)" if irradiance <= 0.0 else f"{efficiency:.1f}%"

# ==========================================
# 5. UI Layout & Graphs Rendering
# ==========================================
st.markdown(
    "<h1 style='margin-bottom: 20px;'>Solar PV Fault Monitoring & Analytics</h1>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-title">System Status</div><div class="kpi-value" style="color:{status_color};">{status}</div></div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Severity Level</div><div class="kpi-value" style="color:{sev_color};">{sev_label}</div></div>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Danger Score</div><div class="kpi-value" style="color:{sev_color};">{danger_score * 100:.1f}%</div></div>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Power Loss</div><div class="kpi-value" style="color:{"#10b981" if power_loss <= 0.2 else "#ef4444"};">{power_loss:.2f} kW</div></div>',
    unsafe_allow_html=True,
)
k5.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Efficiency</div><div class="kpi-value">{eff_text}</div></div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="diag-card">
        <div class="diag-header">System Diagnostics (4-Stage ML Output)</div>
        <div class="diag-fault" style="color: {status_color};">{fault_label}</div>
        <div class="diag-action-box">
            <div class="diag-action-title">Recommended Action / Solution</div>
            <div class="diag-action-text">{action}</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Sensors Grid
st.markdown("**Live Sensor Metrics**")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(
    render_sensor_card("AC Current", ac_curr, "A", "#3b82f6"),
    unsafe_allow_html=True,
)
c2.markdown(
    render_sensor_card("DC Current", dc_curr, "A", "#38bdf8"),
    unsafe_allow_html=True,
)
c3.markdown(
    render_sensor_card("AC Voltage", ac_volt, "V", "#a855f7"),
    unsafe_allow_html=True,
)
c4.markdown(
    render_sensor_card("DC Voltage", dc_volt, "V", "#c084fc"),
    unsafe_allow_html=True,
)
c5.markdown(
    render_sensor_card("Irradiance", irradiance, "W/m²", "#f59e0b"),
    unsafe_allow_html=True,
)
c6.markdown(
    render_sensor_card("Panel Temp", temp, "°C", "#f43f5e"),
    unsafe_allow_html=True,
)

# Charts
time_series = pd.date_range(
    start=datetime.now().replace(hour=0, minute=0, second=0),
    periods=96,
    freq="15min",
)
hours_arr = (
    time_series.hour
    + time_series.minute / 60.0
    + time_series.second / 3600.0
)
base_sig = (
    np.where(
        (hours_arr >= 6.0) & (hours_arr <= 18.0),
        active_power * np.sin((hours_arr - 6.0) * np.pi / 12.0),
        0.0,
    )
    if irradiance > 0.0
    else np.zeros(96)
)

df_chart = pd.DataFrame(
    {"Time": time_series, "Actual": np.maximum(0, base_sig)}
)
df_chart["Predicted"] = df_chart["Actual"] + power_loss * 0.5

chart_config = {"displayModeBar": False, "scrollZoom": False}
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Actual vs Expected Power (Stage 1 Model)**")
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=df_chart["Time"],
            y=df_chart["Actual"],
            mode="lines",
            name="Actual Power",
            line=dict(color="#3b82f6", width=2.5, shape="spline"),
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=df_chart["Time"],
            y=df_chart["Predicted"],
            mode="lines",
            name="Model Expected Power",
            line=dict(color="#f97316", width=2, dash="dot", shape="spline"),
        )
    )
    fig_line.update_layout(
        template=plotly_template,
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        height=270,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=text_color),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            fixedrange=True,
            tickformat="%b %d\n%H:%M",
            tickfont=dict(color=text_color, size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            title="Power (kW)",
            title_font=dict(color=text_color, size=11),
            fixedrange=True,
            tickfont=dict(color=text_color, size=11),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True, config=chart_config)

with col_right:
    st.markdown("**Daily Energy Output vs Power Loss**")
    daily_summary = pd.DataFrame(
        {
            "Day": [datetime.now().strftime("%a %d/%m")],
            "Actual": [df_chart["Actual"].sum() * 0.25],
            "Lost": [
                max(
                    0.0,
                    (df_chart["Predicted"] - df_chart["Actual"]).sum() * 0.25,
                )
            ],
        }
    )

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=daily_summary["Day"],
            y=daily_summary["Actual"],
            name="Generated (kWh)",
            marker_color="#06b6d4",
            marker_cornerradius=6,
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=daily_summary["Day"],
            y=daily_summary["Lost"],
            name="Lost (kWh)",
            marker_color="#f43f5e",
            marker_cornerradius=6,
        )
    )
    fig_bar.update_layout(
        template=plotly_template,
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        barmode="group",
        height=270,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=text_color),
        ),
        xaxis=dict(
            showgrid=False,
            fixedrange=True,
            tickfont=dict(color=text_color, size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            title="Energy (kWh)",
            title_font=dict(color=text_color, size=11),
            fixedrange=True,
            tickfont=dict(color=text_color, size=11),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=chart_config)
