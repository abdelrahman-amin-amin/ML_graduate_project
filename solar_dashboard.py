from datetime import datetime, time
import time as t_lib
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import zipfile
import requests

# ==========================================
# 1. Page Configuration & Dynamic CSS
# ==========================================
st.set_page_config(
    page_title="Solar PV Smart Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Control Panel Title Customization (Enlarged size)
st.sidebar.markdown(
    "<h1 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 10px;'>Control Panel</h1>",
    unsafe_allow_html=True,
)
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
        background-color: {btn_bg}; 
        color: {text_color} !important; 
        padding: 12px 16px !important;
        border-radius: 10px !important; 
        margin: 0 !important; 
        cursor: pointer; 
        width: 100% !important;
        transition: all 0.2s ease; 
        font-weight: 600; 
        font-size: 0.95rem; 
        display: flex !important; 
        align-items: center !important; 
        gap: 12px !important;
        text-align: left !important; 
        border: 1px solid {border_color};
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {active_btn_bg} !important; 
        color: #ffffff !important; 
        border-color: {active_btn_bg} !important; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
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
            <span class="sensor-value" style="color: {color};">{round(float(value), 2)}</span>
            <span class="sensor-unit">{unit}</span>
        </div>
    </div>
    """


# ==========================================
# 2. ML Engine & Models Loading (Blend: CatBoost + LightGBM)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Automatic model download from GitHub Release to handle LFS limitations on deployment servers
RELEASE_ZIP_URL = "https://github.com/abdelrahman-amin-amin/ML_graduate_project/releases/download/v1.0/saved_models.zip"

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    test_model_path = os.path.join(MODELS_DIR, "cat_reg_m1.pkl")
    
    # If the file doesn't exist or is too small (empty LFS pointer), download it automatically
    if not os.path.exists(test_model_path) or os.path.getsize(test_model_path) < 2000:
        with st.spinner("🔄 Downloading AI models, please wait..."):
            try:
                response = requests.get(RELEASE_ZIP_URL)
                if response.status_code == 200:
                    zip_path = os.path.join(BASE_DIR, "saved_models.zip")
                    with open(zip_path, "wb") as f:
                        f.write(response.content)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(BASE_DIR)
                    st.success("✅ Models are ready successfully!")
                else:
                    st.error("❌ Download failed. Make sure you created a Release in your repository and uploaded the zip file with the correct name.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

ensure_models_exist()

REQUIRED_MODELS = {
    "power_cat": os.path.join(MODELS_DIR, "cat_reg_m1.pkl"),
    "power_lgbm": os.path.join(MODELS_DIR, "lgbm_reg_m1.pkl"),
    "power_weight": os.path.join(MODELS_DIR, "blend_weight_m1.pkl"),
    "is_faulted_cat": os.path.join(MODELS_DIR, "cat_model_m2.pkl"),
    "is_faulted_lgbm": os.path.join(MODELS_DIR, "lgbm_model_m2.pkl"),
    "fault_type_cat": os.path.join(MODELS_DIR, "cat_model_m3.pkl"),
    "fault_type_lgbm": os.path.join(MODELS_DIR, "lgbm_model_m3.pkl"),
    "fault_type_classes": os.path.join(MODELS_DIR, "classes_m3.pkl"),
    "danger_cat": os.path.join(MODELS_DIR, "cat_model_m4.pkl"),
    "danger_lgbm": os.path.join(MODELS_DIR, "lgbm_model_m4.pkl"),
    "danger_classes": os.path.join(MODELS_DIR, "classes_m4.pkl"),
}

_STAGE2_3_4_FEATURES = [
    "active_power",
    "irradiance",
    "clear_sky_irradiance",
    "ambient_temp",
    "module_temp",
    "inverter_temp",
    "dc_voltage",
    "dc_current",
    "ac_voltage",
    "ac_current",
    "performance_ratio",
    "cloud_cover",
    "sun_elevation",
    "sun_azimuth",
    "day_length_hours",
    "dc_power",
    "ac_power",
    "inverter_efficiency",
    "hour",
    "month",
    "day_of_week",
    "is_daylight",
]

DEFAULT_STAGE_FEATURES = {
    "power": [
        "irradiance",
        "clear_sky_irradiance",
        "ambient_temp",
        "module_temp",
        "inverter_temp",
        "cloud_cover",
        "sun_elevation",
        "sun_azimuth",
        "day_length_hours",
        "hour",
        "month",
        "day_of_week",
        "is_daylight",
    ],
    "is_faulted": list(_STAGE2_3_4_FEATURES),
    "fault_type": list(_STAGE2_3_4_FEATURES),
    "danger": list(_STAGE2_3_4_FEATURES),
}


@st.cache_resource
def load_models():
    missing_files = [path for path in REQUIRED_MODELS.values() if not os.path.exists(path)]
    if missing_files:
        return None, False, ["Missing file(s):\n" + "\n".join(missing_files)]
    try:
        models = {key: joblib.load(path) for key, path in REQUIRED_MODELS.items()}
        return models, True, []
    except Exception as e:
        return None, False, [str(e)]


models, models_loaded, missing_info = load_models()

if not models_loaded:
    st.error("❌ **Failed to load AI models**")
    st.code(missing_info[0] if missing_info else "Unknown error")
    st.info(f"Expected folder: `{MODELS_DIR}`")
    st.stop()
    raise SystemExit(0)

power_cat = models["power_cat"]
power_lgbm = models["power_lgbm"]
power_weight = models["power_weight"]

is_faulted_cat = models["is_faulted_cat"]
is_faulted_lgbm = models["is_faulted_lgbm"]

fault_type_cat = models["fault_type_cat"]
fault_type_lgbm = models["fault_type_lgbm"]
fault_type_classes = list(models["fault_type_classes"])

danger_cat = models["danger_cat"]
danger_lgbm = models["danger_lgbm"]
danger_classes = list(models["danger_classes"])


# ==========================================
# 2b. Blend Helper Functions
# ==========================================
def blend_regression_predict(cat_model, lgbm_model, features_df, stage_key, weight):
    X_cat = prepare_features(cat_model, features_df, stage_key)
    X_lgbm = prepare_features(lgbm_model, features_df, stage_key)
    pred_cat = cat_model.predict(X_cat)
    pred_lgbm = lgbm_model.predict(X_lgbm)
    return weight * pred_cat + (1 - weight) * pred_lgbm


def blend_binary_proba(cat_model, lgbm_model, features_df, stage_key):
    X_cat = prepare_features(cat_model, features_df, stage_key)
    X_lgbm = prepare_features(lgbm_model, features_df, stage_key)
    proba_cat = cat_model.predict_proba(X_cat)[:, 1]
    proba_lgb = lgbm_model.predict_proba(X_lgbm)[:, 1]
    return (proba_cat + proba_lgb) / 2


def blend_multiclass_proba(cat_model, lgb_model, features_df, stage_key, classes_order):
    X_cat = prepare_features(cat_model, features_df, stage_key)
    X_lgb = prepare_features(lgb_model, features_df, stage_key)
    proba_cat = cat_model.predict_proba(X_cat)
    proba_lgb = lgb_model.predict_proba(X_lgb)

    cat_classes = list(cat_model.classes_.ravel())
    lgb_classes = list(lgb_model.classes_)

    proba_cat_df = pd.DataFrame(proba_cat, columns=cat_classes)[classes_order]
    proba_lgb_df = pd.DataFrame(proba_lgb, columns=lgb_classes)[classes_order]

    proba_blend = (proba_cat_df.values + proba_lgb_df.values) / 2
    pred_blend = (
        pd.Series(proba_blend.argmax(axis=1))
        .map(dict(enumerate(classes_order)))
        .values
    )
    return pred_blend, proba_blend


def compute_danger_score(proba_row, classes_order):
    severity_rank = {
        "none": 0, "healthy": 0, "0": 0, "false": 0, "no": 0,
        "low": 1, "1": 1,
        "medium": 2, "moderate": 2, "2": 2,
        "high": 3, "critical": 3, "severe": 3, "3": 3,
    }

    ranks = []
    for cls in classes_order:
        key = str(cls).strip().lower()
        if key in severity_rank:
            ranks.append(severity_rank[key])
        else:
            try:
                ranks.append(float(cls))
            except (TypeError, ValueError):
                ranks.append(3)

    ranks = np.asarray(ranks, dtype=float)
    max_rank = ranks.max() if ranks.max() > 0 else 1.0
    proba_row = np.asarray(proba_row, dtype=float)
    return float(np.clip(np.sum(proba_row * (ranks / max_rank)), 0.0, 1.0))


def compute_features_from_row(row):
    time_val = row.get("Timestamp", row.get("time", datetime.now()))
    dt = (
        pd.to_datetime(time_val)
        if not isinstance(time_val, pd.Timestamp)
        else time_val
    )

    hour = float(dt.hour)
    month = dt.month
    day_of_week = dt.dayofweek

    irradiance = float(row.get("irradiance", 0.0))
    temp = float(row.get("module_temp", row.get("temp", 25.0)))
    ac_curr = float(row.get("ac_current", 0.0))
    dc_curr = float(row.get("dc_current", 0.0))
    ac_volt = float(row.get("ac_voltage", 0.0))
    dc_volt = float(row.get("dc_voltage", 0.0))

    is_daylight = 1 if (6 <= dt.hour <= 18 or irradiance > 15.0) else 0
    
    if is_daylight == 0:
        irradiance = 0.0
        ac_curr = 0.0
        dc_curr = 0.0
        active_power = 0.0
        dc_power = 0.0
        ac_power = 0.0
        performance_ratio = 0.0
        inverter_efficiency = 0.0
    else:
        active_power = max(
            0.0,
            float(
                row.get(
                    "active_power",
                    (ac_volt * ac_curr)
                    if ac_volt > 0 and ac_curr > 0
                    else 0.0,
                )
            ),
        )

        dc_power = float(row.get("dc_power", dc_volt * dc_curr))
        ac_power = float(row.get("ac_power", ac_volt * ac_curr))

        if "inverter_efficiency" in row and pd.notna(row.get("inverter_efficiency")):
            inverter_efficiency = float(row["inverter_efficiency"])
        else:
            inverter_efficiency = (ac_power / dc_power) if dc_power > 0 else 0.0
        inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))

        if "performance_ratio" in row and pd.notna(row.get("performance_ratio")):
            performance_ratio = float(row["performance_ratio"])
        else:
            performance_ratio = (
                min(1.2, max(0.0, active_power / ((irradiance / 1000.0) * 4.0)))
                if irradiance > 50.0
                else 0.0
            )

    day_length_hours = float(row.get("day_length_hours", 12.0))

    if "sun_elevation" in row and pd.notna(row.get("sun_elevation")):
        sun_elevation = float(row["sun_elevation"])
    else:
        sun_elevation = max(
            0.0, 90.0 * np.sin(np.pi * (hour - 6) / 12.0) if is_daylight else 0.0
        )

    if "sun_azimuth" in row and pd.notna(row.get("sun_azimuth")):
        sun_azimuth = float(row["sun_azimuth"])
    else:
        sun_azimuth = (hour / 24.0) * 360.0

    if "ambient_temp" in row and pd.notna(row.get("ambient_temp")):
        amb_temp = float(row["ambient_temp"])
    else:
        amb_temp = max(-5.0, temp - 8.0)

    if "clear_sky_irradiance" in row and pd.notna(row.get("clear_sky_irradiance")):
        clear_sky_irradiance = float(row["clear_sky_irradiance"])
    else:
        clear_sky_irradiance = max(
            0.0, 1000.0 * np.sin(np.pi * (hour - 6) / 12.0) if is_daylight else 0.0
        )

    cloud_cover = float(row.get("cloud_cover", 0.1))

    module_temp = temp
    inverter_temp = float(row.get("inverter_temp", temp * 0.85 + 5.0))

    return pd.DataFrame(
        [
            {
                "timestamp": dt,
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
                "ac_power": float(ac_power),
                "dc_voltage": float(dc_volt),
                "dc_current": float(dc_curr),
                "ac_voltage": float(ac_volt),
                "ac_current": float(ac_curr),
                "module_temp": float(module_temp),
                "inverter_temp": float(inverter_temp),
                "performance_ratio": float(performance_ratio),
                "inverter_efficiency": float(inverter_efficiency),
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


def scale_expected_power(raw_pred, active_power):
    val = max(0.0, float(raw_pred))
    return max(active_power, val)


# ==========================================
# 3. Sidebar & Inputs
# ==========================================
data_source = st.sidebar.radio(
    "Data Source",
    ["Live Feed", "Manual Entry", "Upload CSV"],
    index=2,
)
st.sidebar.markdown(
    "<hr style='margin: 15px 0; opacity: 0.15;'>", unsafe_allow_html=True
)

st.sidebar.markdown("### Model Debug")
fault_threshold = st.sidebar.slider(
    "Fault Detection Threshold", 0.05, 0.95, 0.50, 0.05
)
POWER_LOSS_ALERT = st.sidebar.slider(
    "Power Loss Alert Level (raw power units)", 10.0, 2000.0, 300.0, 10.0,
    help="If actual power falls below expected power by at least this much, "
         "flag it as a Performance Anomaly even if the fault model says healthy.",
)
show_debug = st.sidebar.checkbox("Show raw model outputs", value=False)

df_uploaded = None

if data_source == "Live Feed":
    row_data = {
        "irradiance": 850.0,
        "module_temp": 42.0,
        "ac_current": 6.0,
        "dc_current": 5.4,
        "ac_voltage": 390.0,
        "dc_voltage": 720.0,
        "active_power": 2340.0,
        "Timestamp": datetime.now(),
    }
elif data_source == "Manual Entry":
    st.sidebar.markdown("### Manual Inputs")
    row_data = {
        "irradiance": st.sidebar.slider(
            "Irradiance (W/m²)", 0.0, 1200.0, 790.0, 10.0
        ),
        "module_temp": st.sidebar.slider("Panel Temp (°C)", 0.0, 80.0, 49.0, 1.0),
        "ac_current": st.sidebar.slider("AC Current (A)", 0.0, 7.0, 3.5, 0.1),
        "dc_current": st.sidebar.slider("DC Current (A)", 0.0, 6.0, 3.0, 0.1),
        "ac_voltage": st.sidebar.slider("AC Voltage (V)", 0.0, 410.0, 350.0, 5.0),
        "dc_voltage": st.sidebar.slider("DC Voltage (V)", 0.0, 840.0, 600.0, 5.0),
        "Timestamp": datetime.now(),
    }
else:
    st.sidebar.markdown("### Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        play_stream = st.sidebar.checkbox(
            "Play Timeline Minute-by-Minute", value=False
        )

        if play_stream:
            row_idx = st.sidebar.slider(
                "Playback Step", 0, len(df_uploaded) - 1, 0, 1
            )
        else:
            row_idx = st.sidebar.slider(
                "Select Row Index", 0, len(df_uploaded) - 1, 0, 1
            )

        row_data = df_uploaded.iloc[row_idx].to_dict()
    else:
        st.sidebar.info("Please upload a CSV file to view data.")
        row_data = {
            "irradiance": 790.0,
            "module_temp": 49.0,
            "ac_current": 3.5,
            "dc_current": 3.2,
            "ac_voltage": 350.0,
            "dc_voltage": 600.0,
            "Timestamp": datetime.now(),
        }

features_df = compute_features_from_row(row_data)
active_power = features_df["active_power"].values[0]
irradiance = features_df["irradiance"].values[0]

# ==========================================
# 4. Sequential Execution (Diagnostics)
# ==========================================
fault_solutions = {
    "soiling": (
        "Schedule a panel surface cleaning to remove dust, dirt, or bird"
        " droppings accumulating on the panels and reducing efficiency."
    ),
    "tracker_stuck": (
        "Inspect the solar tracker's motor, actuator, and control unit — the"
        " tracker appears stuck and is no longer following the sun's path."
    ),
    "dc_string_outage": (
        "Inspect DC wiring, junction box connectors, and string fuses for a"
        " disconnected, broken, or open PV string."
    ),
    "inverter_overheat": (
        "Check inverter cooling fans, ventilation paths, and ambient airflow;"
        " consider derating output until inverter temperature drops."
    ),
    "downtime": (
        "System is reporting unexpected downtime; check inverter power and"
        " communication status as well as the grid connection."
    ),
}

FAULT_TYPE_DISPLAY_NAMES = {
    "healthy": "Healthy",
    "soiling": "Soiling",
    "tracker_stuck": "Tracker Stuck",
    "dc_string_outage": "DC String Outage",
    "inverter_overheat": "Inverter Overheat",
    "downtime": "Downtime",
}

if irradiance <= 15.0:
    expected_power = 0.0
    power_loss = 0.0
    is_faulted = 0
    raw_type_pred = "healthy"
    danger_score = 0.0
    sev_label, sev_color = "None", "#64748b"
    status, status_color = "Night / Low Sun", "#64748b"
    fault_label = "Normal Inactive State (Night/Low Irradiance)"
    action = "System is inactive or in low-light conditions. No generation expected, no faults detected."
else:
    raw_exp_power = float(
        blend_regression_predict(power_cat, power_lgbm, features_df, "power", power_weight)[0]
    )
    expected_power = scale_expected_power(raw_exp_power, active_power)
    power_loss = max(0.0, round(expected_power - active_power, 2))

    proba_faulted_blend = blend_binary_proba(is_faulted_cat, is_faulted_lgbm, features_df, "is_faulted")
    is_faulted = int(proba_faulted_blend[0] >= fault_threshold)

    if is_faulted == 1:
        _, proba_type_blend = blend_multiclass_proba(
            fault_type_cat, fault_type_lgbm, features_df, "fault_type", fault_type_classes
        )
        
        _, proba_danger_blend = blend_multiclass_proba(
            danger_cat, danger_lgbm, features_df, "danger", danger_classes
        )
        danger_score = compute_danger_score(proba_danger_blend[0], danger_classes)

        type_probs = dict(zip(fault_type_classes, proba_type_blend[0]))
        filtered_types = {k: v for k, v in type_probs.items() if str(k).lower() != "healthy"}
        
        if filtered_types:
            raw_type_pred = max(filtered_types, key=filtered_types.get)
        else:
            raw_type_pred = max(type_probs, key=type_probs.get)
    else:
        raw_type_pred = "healthy"
        danger_score = 0.0

    is_flagged_by_models = is_faulted == 1
    is_flagged_by_power_loss = power_loss >= POWER_LOSS_ALERT

    if is_flagged_by_models:
        status, status_color = "Fault Detected", "#ef4444"
        
        fault_label = FAULT_TYPE_DISPLAY_NAMES.get(
            str(raw_type_pred), str(raw_type_pred).replace("_", " ").title()
        )
            
        if danger_score >= 0.75:
            sev_label, sev_color = "Critical", "#dc2626"
        elif danger_score >= 0.45:
            sev_label, sev_color = "High", "#ef4444"
        elif danger_score >= 0.20:
            sev_label, sev_color = "Medium", "#f59e0b"
        else:
            sev_label, sev_color = "Low", "#10b981"

        action = fault_solutions.get(
            str(raw_type_pred),
            "Perform a comprehensive on-site inspection of the PV strings, check"
            " electrical connections, and review inverter status.",
        )
    elif is_flagged_by_power_loss:
        status, status_color = "Performance Anomaly", "#f59e0b"
        sev_label, sev_color = "Medium", "#f59e0b"
        danger_score = max(danger_score, 0.25)
        fault_label = "Unexplained Power Loss (Performance Drop)"
        action = (
            f"Actual power is {power_loss:.2f} kW below the model's expected power, "
            "but the fault-detection model did not cross its confidence threshold. "
            "Worth a manual check: irradiance sensor accuracy, partial shading, or "
            "soiling that the model may be under-confident about."
        )
    else:
        status, status_color = "Healthy", "#10b981"
        sev_label, sev_color = "Low", "#10b981"
        danger_score = min(danger_score, 0.15)
        fault_label = "Healthy System (Normal)"
        action = "System operates within optimal parameters. Regular monitoring is sufficient; no immediate physical inspection required."

efficiency = features_df["inverter_efficiency"].values[0]
eff_text = "N/A (Night)" if irradiance <= 15.0 else f"{efficiency*100:.1f}%"

if show_debug:
    with st.expander("🔧 Raw model outputs (debug)", expanded=True):
        st.write("**Computed features fed to the models:**")
        st.dataframe(features_df)
        if irradiance > 15.0:
            st.write(f"**is_faulted blend probability:** `{proba_faulted_blend[0]:.4f}`  (threshold = `{fault_threshold}`)")
            st.write(f"**raw expected_power (blend, unscaled):** `{raw_exp_power:.4f}`")
            st.write(f"**expected_power (final):** `{expected_power:.4f}` kW  |  **active_power:** `{active_power:.4f}` kW  |  **power_loss:** `{power_loss:.4f}` kW")
            st.write(f"**raw_type_pred (blend):** `{raw_type_pred}`")
            st.write(f"**danger_score (blend):** `{danger_score:.4f}`")
        else:
            st.info("Irradiance ≤ 15 W/m² → night/low-sun branch, models are skipped entirely.")

# ==========================================
# 5. UI Layout & KPI Cards
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
    f'<div class="kpi-card"><div class="kpi-title">Power Loss</div><div class="kpi-value" style="color:{"#10b981" if power_loss <= 15.0 else "#ef4444"};">{power_loss:.1f}</div></div>',
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

st.markdown("**Live Sensor Metrics (Selected Row)**")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(
    render_sensor_card(
        "AC Current", row_data.get("ac_current", 0), "A", "#3b82f6"
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    render_sensor_card(
        "DC Current", row_data.get("dc_current", 0), "A", "#38bdf8"
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    render_sensor_card(
        "AC Voltage", row_data.get("ac_voltage", 0), "V", "#a855f7"
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    render_sensor_card(
        "DC Voltage", row_data.get("dc_voltage", 0), "V", "#c084fc"
    ),
    unsafe_allow_html=True,
)
c5.markdown(
    render_sensor_card(
        "Irradiance", row_data.get("irradiance", 0), "W/m²", "#f59e0b"
    ),
    unsafe_allow_html=True,
)
c6.markdown(
    render_sensor_card(
        "Panel Temp", row_data.get("module_temp", 0), "°C", "#f43f5e"
    ),
    unsafe_allow_html=True,
)

# ==========================================
# 6. Charts Generation
# ==========================================
chart_config = {"displayModeBar": False, "scrollZoom": False}
col_left, col_right = st.columns(2)

if df_uploaded is not None and (
    "Timestamp" in df_uploaded.columns or "time" in df_uploaded.columns
):
    time_col = "Timestamp" if "Timestamp" in df_uploaded.columns else "time"
    df_chart_full = df_uploaded.copy()
    df_chart_full["ParsedTime"] = pd.to_datetime(df_chart_full[time_col])
    df_chart_full = df_chart_full.sort_values("ParsedTime")

    df_chart_full["Actual_Power"] = df_chart_full.get(
        "active_power",
        (
            df_chart_full.get("ac_voltage", 220)
            * df_chart_full.get("ac_current", 10)
        )
        / 1000.0,
    )
    df_chart_full["Expected_Power"] = df_chart_full["Actual_Power"] * 1.05

    if locals().get("play_stream", False):
        df_plot_subset = df_chart_full.iloc[: row_idx + 1]
    else:
        df_plot_subset = df_chart_full

    with col_left:
        st.markdown(
            "**Actual vs Expected Power (Progressive Timeline)**"
            if locals().get("play_stream", False)
            else "**Actual vs Expected Power (Full Timeline)**"
        )
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=df_plot_subset["ParsedTime"],
                y=df_plot_subset["Actual_Power"],
                mode="lines",
                name="Actual Power",
                line=dict(color="#3b82f6", width=2, shape="spline"),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=df_plot_subset["ParsedTime"],
                y=df_plot_subset["Expected_Power"],
                mode="lines",
                name="Expected Power",
                line=dict(color="#f97316", width=1.5, dash="dot"),
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
        st.markdown("**Daily Energy Output vs Power Loss (All Days)**")
        df_chart_full["Date"] = df_chart_full["ParsedTime"].dt.date
        df_chart_full["Generated_Energy"] = (
            df_chart_full["Actual_Power"] * 1.0
        )
        df_chart_full["Lost_Energy"] = 0.05 * df_chart_full["Generated_Energy"]

        daily_grouped = (
            df_chart_full.groupby("Date")[
                ["Generated_Energy", "Lost_Energy"]
            ]
            .sum()
            .reset_index()
        )

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=daily_grouped["Date"].astype(str),
                y=daily_grouped["Generated_Energy"],
                name="Generated (kWh)",
                marker_color="#06b6d4",
                marker_cornerradius=6,
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=daily_grouped["Date"].astype(str),
                y=daily_grouped["Lost_Energy"],
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
                tickfont=dict(color=text_color, size=10),
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

    if locals().get("play_stream", False) and row_idx < len(df_uploaded) - 1:
        t_lib.sleep(0.3)
        st.rerun()

else:
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
    base_sig = np.where(
        (hours_arr >= 6.0) & (hours_arr <= 18.0),
        active_power * np.sin((hours_arr - 6.0) * np.pi / 12.0),
        0.0,
    )
    df_chart = pd.DataFrame(
        {"Time": time_series, "Actual": np.maximum(0, base_sig)}
    )
    df_chart["Predicted"] = df_chart["Actual"] + power_loss * 0.4

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
                        (df_chart["Predicted"] - df_chart["Actual"]).sum()
                        * 0.25,
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
