from datetime import datetime, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Solar PV Smart Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sidebar Control & Theme Toggle
st.sidebar.title("Control Panel")

# Dark Mode Toggle
dark_mode = st.sidebar.toggle("Dark Mode", value=True)

# Dynamic Color Variables
if dark_mode:
    bg_app = "#0b0f19"
    card_bg = "#151c2c"
    text_color = "#f8fafc"
    subtext_color = "#94a3b8"
    border_color = "#1e293b"
    plotly_template = "plotly_dark"
    grid_color = "#1e293b"
else:
    bg_app = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#0f172a"
    subtext_color = "#64748b"
    border_color = "#e2e8f0"
    plotly_template = "plotly_white"
    grid_color = "#e2e8f0"

# 3. Custom CSS Styles
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_app};
        color: {text_color};
    }}
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    .block-container {{
        padding-top: 2.5rem !important;
        padding-bottom: 1.5rem;
    }}
    .sensor-card {{
        background: {card_bg};
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid {border_color};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
        transition: transform 0.2s ease;
    }}
    .sensor-card:hover {{
        transform: translateY(-2px);
    }}
    .sensor-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .sensor-title {{
        color: {subtext_color};
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .sensor-body {{
        display: flex;
        align-items: baseline;
        gap: 6px;
    }}
    .sensor-value {{
        font-size: 1.35rem;
        font-weight: 700;
    }}
    .sensor-unit {{
        color: {subtext_color};
        font-size: 0.85rem;
        font-weight: 500;
    }}
    .kpi-card {{
        background: {card_bg};
        border-radius: 12px;
        padding: 12px;
        border: 1px solid {border_color};
        text-align: center;
    }}
    .kpi-title {{ color: {subtext_color}; font-size: 0.78rem; font-weight: 600; }}
    .kpi-value {{ font-size: 1.2rem; font-weight: 700; margin-top: 2px; color: {text_color}; }}
    </style>
""",
    unsafe_allow_html=True,
)


def render_sensor_card(title, value, unit, color="#3b82f6"):
    return f"""
    <div class="sensor-card">
        <div class="sensor-header">
            <span class="sensor-title">{title}</span>
        </div>
        <div class="sensor-body">
            <span class="sensor-value" style="color: {color};">{round(value, 2)}</span>
            <span class="sensor-unit">{unit}</span>
        </div>
    </div>
    """


def generate_realistic_solar_profile(time_series, max_power):
    """دالة حساب منحنى التوليد الشمسي الواقعي"""
    ts = (
        pd.Series(time_series)
        if isinstance(time_series, pd.DatetimeIndex)
        else time_series
    )
    hours = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    solar_power = np.where(
        (hours >= 6.0) & (hours <= 18.0),
        max_power * np.sin((hours - 6.0) * np.pi / 12.0),
        0.0,
    )
    return solar_power


# 4. Data Source Settings
data_source = st.sidebar.radio(
    "Data Source:", ["Live Feed", "Manual Entry", "Upload CSV"]
)

# Default Metric Values (تنسيق قيم التيار والجهد لتوفير كفاءة ~95.4%)
ac_curr, dc_curr = 14.3, 9.2
ac_volt, dc_volt = 230.0, 375.0
irradiance, temp = 850.0, 42.0

df_chart = pd.DataFrame()

# Mode A: Manual Entry
if data_source == "Manual Entry":
    st.sidebar.subheader("Sensor Inputs")
    irradiance = float(st.sidebar.slider("Irradiance (W/m²)", 0, 1200, 850))
    temp = float(st.sidebar.slider("Panel Temp (°C)", -10, 85, 42))

    irr_factor = irradiance / 1000.0

    # القيم المحسوبة تلقائياً تحافظ على كفاءة الـ Inverter العالية
    dc_curr = st.sidebar.number_input(
        "DC Current (A)", value=round(9.2 * irr_factor, 2), step=0.1
    )
    ac_curr = st.sidebar.number_input(
        "AC Current (A)", value=round(14.3 * irr_factor, 2), step=0.1
    )
    dc_volt = st.sidebar.number_input(
        "DC Voltage (V)", value=375.0 if irradiance > 0 else 0.0, step=1.0
    )
    ac_volt = st.sidebar.number_input(
        "AC Voltage (V)", value=230.0 if irradiance > 0 else 0.0, step=1.0
    )

    st.sidebar.subheader("Time Settings")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now())
    start_time = datetime.combine(start_date, time(0, 0))
    interval_min = st.sidebar.selectbox(
        "Logging Interval (Minutes)", [15, 30, 60], index=0
    )

    time_series = pd.date_range(
        start=start_time, periods=96, freq=f"{interval_min}min"
    )

    p_peak = (ac_volt * ac_curr) / 1000.0 if irradiance > 0 else 0.0
    base_signal = generate_realistic_solar_profile(time_series, p_peak)

    np.random.seed(42)
    noise = np.where(
        base_signal > 0, np.random.normal(0, 0.03, len(time_series)), 0.0
    )

    df_chart = pd.DataFrame(
        {
            "Time": time_series,
            "Actual": np.maximum(0, base_signal + noise),
            "Predicted": np.maximum(0, (base_signal * 1.02) + noise),
        }
    )

# Mode B: CSV File Upload
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Metrics File", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            cols = {str(c).lower().strip(): c for c in df_raw.columns}

            last_row = df_raw.iloc[-1]

            def extract_val(possible_keys, default_val):
                for k in possible_keys:
                    if k in cols:
                        return float(last_row[cols[k]])
                return default_val

            ac_curr = extract_val(
                ["ac_current", "ac_curr", "ac_i", "i_ac"], ac_curr
            )
            dc_curr = extract_val(
                ["dc_current", "dc_curr", "dc_i", "i_dc"], dc_curr
            )
            ac_volt = extract_val(
                ["ac_voltage", "ac_volt", "ac_v", "v_ac"], ac_volt
            )
            dc_volt = extract_val(
                ["dc_voltage", "dc_volt", "dc_v", "v_dc"], dc_volt
            )
            irradiance = extract_val(
                ["irradiance", "irr", "poa", "ghi"], irradiance
            )
            temp = extract_val(
                [
                    "module_temp",
                    "ambient_temp",
                    "temp",
                    "temperature",
                    "panel_temp",
                ],
                temp,
            )

            time_col = next(
                (
                    cols[k]
                    for k in ["time", "date", "timestamp", "datetime"]
                    if k in cols
                ),
                None,
            )
            act_col = next(
                (
                    cols[k]
                    for k in [
                        "active_power",
                        "actual",
                        "actual_power",
                        "p_actual",
                        "power",
                        "i_ac",
                    ]
                    if k in cols
                ),
                None,
            )
            pred_col = next(
                (
                    cols[k]
                    for k in [
                        "predicted",
                        "predicted_power",
                        "p_pred",
                        "predict",
                    ]
                    if k in cols
                ),
                None,
            )

            df_chart = pd.DataFrame()
            if time_col:
                df_chart["Time"] = pd.to_datetime(df_raw[time_col])
            else:
                df_chart["Time"] = pd.date_range(
                    end=pd.Timestamp.now(), periods=len(df_raw), freq="15min"
                )

            if act_col:
                df_chart["Actual"] = (
                    df_raw[act_col] / 1000.0
                    if df_raw[act_col].max() > 100
                    else df_raw[act_col]
                )
            else:
                df_chart["Actual"] = generate_realistic_solar_profile(
                    df_chart["Time"], (ac_volt * ac_curr / 1000.0)
                )

            if pred_col:
                df_chart["Predicted"] = (
                    df_raw[pred_col] / 1000.0
                    if df_raw[pred_col].max() > 100
                    else df_raw[pred_col]
                )
            else:
                df_chart["Predicted"] = df_chart["Actual"] * 1.02

            st.sidebar.success("CSV File Loaded Successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# Mode C: Live Feed Simulation Default
if df_chart.empty:
    time_series = pd.date_range(
        start=datetime.now().replace(hour=0, minute=0, second=0),
        periods=96,
        freq="15min",
    )
    base_signal = generate_realistic_solar_profile(time_series, max_power=3.29)

    np.random.seed(42)
    noise = np.where(
        base_signal > 0, np.random.normal(0, 0.03, len(time_series)), 0.0
    )

    df_chart = pd.DataFrame(
        {
            "Time": time_series,
            "Actual": np.maximum(0, base_signal + noise),
            "Predicted": np.maximum(0, (base_signal * 1.02) + noise),
        }
    )

# 5. Dynamic Metrics & Night Standby Detection Logic
p_dc_kw = (dc_volt * dc_curr) / 1000.0
p_ac_kw = (ac_volt * ac_curr) / 1000.0
power_loss_kw = max(0.0, p_dc_kw - p_ac_kw)
efficiency = (p_ac_kw / p_dc_kw * 100.0) if p_dc_kw > 0 else 0.0

is_night = (irradiance <= 10.0) or (p_dc_kw == 0.0 and p_ac_kw == 0.0)

if is_night:
    status_label, status_color = "Night Standby", "#94a3b8"
    severity_label, severity_color = "None", "#94a3b8"
    efficiency_text = "N/A (Night)"
elif efficiency >= 85.0:
    status_label, status_color = "Normal", "#10b981"
    severity_label, severity_color = "Low", "#10b981"
    efficiency_text = f"{efficiency:.1f}%"
elif efficiency >= 70.0:
    status_label, status_color = "Warning", "#f59e0b"
    severity_label, severity_color = "Medium", "#f59e0b"
    efficiency_text = f"{efficiency:.1f}%"
else:
    status_label, status_color = "Fault Detected", "#ef4444"
    severity_label, severity_color = "High", "#ef4444"
    efficiency_text = f"{efficiency:.1f}%"

# 6. Title and Top KPI Cards
st.markdown(
    "<h1 style='margin-bottom: 20px;'>Solar PV Fault Monitoring & Analytics</h1>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-title">System Status</div><div class="kpi-value" style="color:{status_color};">{status_label}</div></div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Severity Level</div><div class="kpi-value" style="color:{severity_color};">{severity_label}</div></div>',
    unsafe_allow_html=True,
)
k3.markdown(
    '<div class="kpi-card"><div class="kpi-title">Prediction Conf.</div><div class="kpi-value">98.4%</div></div>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Power Loss</div><div class="kpi-value" style="color:{"#10b981" if power_loss_kw < 0.3 else "#ef4444"};">{power_loss_kw:.2f} kW</div></div>',
    unsafe_allow_html=True,
)
k5.markdown(
    f'<div class="kpi-card"><div class="kpi-title">Efficiency</div><div class="kpi-value">{efficiency_text}</div></div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Live Sensor Cards
st.markdown("**Live Sensor Metrics**")
col_s1, col_s2, col_s3, col_s4, col_s5, col_s6 = st.columns(6)

with col_s1:
    st.markdown(
        render_sensor_card("AC Current", ac_curr, "A", "#3b82f6"),
        unsafe_allow_html=True,
    )
with col_s2:
    st.markdown(
        render_sensor_card("DC Current", dc_curr, "A", "#38bdf8"),
        unsafe_allow_html=True,
    )
with col_s3:
    st.markdown(
        render_sensor_card("AC Voltage", ac_volt, "V", "#a855f7"),
        unsafe_allow_html=True,
    )
with col_s4:
    st.markdown(
        render_sensor_card("DC Voltage", dc_volt, "V", "#c084fc"),
        unsafe_allow_html=True,
    )
with col_s5:
    st.markdown(
        render_sensor_card("Irradiance", irradiance, "W/m²", "#f59e0b"),
        unsafe_allow_html=True,
    )
with col_s6:
    st.markdown(
        render_sensor_card("Panel Temp", temp, "°C", "#f43f5e"),
        unsafe_allow_html=True,
    )

# 8. Interactive Visualizations
chart_config = {"displayModeBar": False, "scrollZoom": False}
col_left, col_right = st.columns(2)

# Chart A: Line Chart
with col_left:
    st.markdown("**Actual vs Predicted Power Output**")
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
            name="Predicted Power",
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

# Chart B: Bar Chart (Daily Aggregated)
with col_right:
    st.markdown("**Generation vs Lost Power (Daily Aggregated)**")

    df_chart["Day"] = df_chart["Time"].dt.strftime("%a %d/%m")

    daily_summary = (
        df_chart.groupby("Day", sort=False)[["Actual", "Predicted"]]
        .sum()
        .reset_index()
    )

    daily_summary["Lost"] = np.maximum(
        0, daily_summary["Predicted"] - daily_summary["Actual"]
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