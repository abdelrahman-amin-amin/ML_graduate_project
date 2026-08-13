import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"

DATA_URL = "https://drive.google.com/uc?export=download&id=1Q0LZS2n7tEe-_H0o7vQsn8jbheRlFajU"

st.set_page_config(
    page_title="Solar AI Command Center",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= DATA SOURCE =========================
uploaded_file = st.sidebar.file_uploader(
    "Upload Data (CSV)",
    type=["csv"],
    help="Upload a CSV file to use instead of the default Google Drive dataset."
)


# ========================= THEME ============================
if "dark" not in st.session_state:
    st.session_state.dark = True

DARK = st.session_state.dark

if DARK:
    BG, PANEL, PANEL2, TEXT, MUTED, BORDER, GRID = (
        "#070B12", "#0D141F", "#111C2B", "#F7FAFC",
        "#91A4B8", "#1E2C3D", "#1B2A3A"
    )
    ACCENT = "#38BDF8"
else:
    BG, PANEL, PANEL2, TEXT, MUTED, BORDER, GRID = (
        "#F4F7FB", "#FFFFFF", "#EEF3F8", "#132033",
        "#66768A", "#D9E2EC", "#DCE5EE"
    )
    ACCENT = "#0284C7"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.stApp {{ background:{BG}; color:{TEXT}; }}
[data-testid="stSidebar"] {{
    background:{PANEL};
    border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1rem; }}
.block-container {{ max-width: 1650px; padding: 1.2rem 2rem 2rem; }}
h1,h2,h3,h4,h5,p,span,label {{ color:{TEXT}; }}
.small-muted {{ color:{MUTED}; font-size:.82rem; }}
.brand {{
    display:flex; align-items:center; gap:12px; padding:8px 4px 16px;
}}
.brand-icon {{
    width:42px;height:42px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#0EA5E9,#14B8A6);
    color:white;font-size:22px;font-weight:900;
}}
.brand-title {{ font-weight:850; font-size:1.05rem; line-height:1.05; }}
.brand-sub {{ color:{MUTED}; font-size:.70rem; margin-top:4px; }}
.section-label {{
    color:{MUTED}; font-size:.68rem; letter-spacing:.13em;
    font-weight:800; margin:18px 4px 8px; text-transform:uppercase;
}}
.hero {{
    border:1px solid {BORDER}; border-radius:22px; padding:26px 28px;
    background:linear-gradient(120deg,#0B1726,#0C2940,#083D43);
    box-shadow:0 18px 60px rgba(0,0,0,.18);
}}
.hero-kicker {{ color:#7DD3FC; font-weight:800; font-size:.72rem; letter-spacing:.14em; }}
.hero-title {{ color:white; font-size:2.15rem; font-weight:900; margin:5px 0; }}
.hero-text {{ color:#C7D8E8; max-width:900px; }}
.status {{
    display:inline-flex; align-items:center; gap:7px; padding:6px 10px;
    border-radius:999px; background:rgba(34,197,94,.12);
    border:1px solid rgba(34,197,94,.28); color:#86EFAC;
    font-size:.75rem; font-weight:800;
}}
.kpi {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:18px;
    padding:17px 18px; min-height:118px;
}}
.kpi-label {{ color:{MUTED}; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; }}
.kpi-value {{ color:{TEXT}; font-size:1.8rem; font-weight:900; margin-top:8px; }}
.kpi-foot {{ color:{MUTED}; font-size:.75rem; margin-top:4px; }}
.card {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:18px; padding:18px;
}}
.card-title {{ font-size:1rem; font-weight:850; color:{TEXT}; }}
.card-sub {{ color:{MUTED}; font-size:.78rem; margin-top:2px; }}
.control {{
    background:{PANEL2}; border:1px solid {BORDER}; border-radius:14px; padding:12px;
}}
div[data-testid="stMetric"] {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:12px;
}}
.stButton > button {{
    border-radius:11px; border:1px solid {BORDER};
    background:{PANEL2}; color:{TEXT}; font-weight:750;
}}
.stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background:{PANEL}; border-color:{BORDER};
}}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:14px; }}
hr {{ border-color:{BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ========================= HELPERS ==========================
def card(title, subtitle=""):
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div>'
        f'<div class="card-sub">{subtitle}</div></div>',
        unsafe_allow_html=True
    )

def kpi(label, value, foot=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
        unsafe_allow_html=True
    )

def chart(fig, height=360):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
        margin=dict(l=8,r=8,t=45,b=8),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_data
def load_uploaded_data(file):
    return pd.read_csv(file)

@st.cache_data
def prepare(df):
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], errors="coerce")
        d = d.sort_values("time")
        d["hour"] = d["time"].dt.hour
        d["month"] = d["time"].dt.month
        d["week"] = d["time"].dt.isocalendar().week.astype(int)
        d["day_of_week"] = d["time"].dt.dayofweek
        d["is_daylight"] = d["hour"].between(6,18).astype(int)
    if "dc_voltage" in d.columns and "dc_current" in d.columns:
        d["dc_power"] = d["dc_voltage"] * d["dc_current"]
    if "ac_voltage" in d.columns and "ac_current" in d.columns:
        d["ac_power"] = d["ac_voltage"] * d["ac_current"]
    if "dc_power" in d.columns and "ac_power" in d.columns:
        d["inverter_efficiency"] = np.where(
            d["dc_power"] > 0, d["ac_power"]/d["dc_power"], 0.0
        ).clip(0,1)
    if "fault_severity" in d.columns and d["fault_severity"].dtype == object:
        d["fault_severity"] = d["fault_severity"].map(
            {"none":0,"low":1,"medium":2,"high":3}
        )
    return d

@st.cache_resource
def load_models():
    # Actual model artifact names in the GitHub models/ folder.
    model_files = {
        "Power / CatBoost": MODEL_DIR / "cat_reg_m1.joblib",
        "Power / LightGBM": MODEL_DIR / "lgbm_reg_m1.joblib",

        "Fault / CatBoost": MODEL_DIR / "cat_model_m2.joblib",
        "Fault / LightGBM": MODEL_DIR / "lgbm_model_m2.joblib",

        "Cause / CatBoost": MODEL_DIR / "cat_model_m3.joblib",
        "Cause / LightGBM": MODEL_DIR / "lgbm_model_m3.joblib",

        "Severity / CatBoost": MODEL_DIR / "cat_model_m4.joblib",
        "Severity / LightGBM": MODEL_DIR / "lgbm_model_m4.joblib",
    }

    models = {}
    missing = []

    for name, path in model_files.items():
        if not path.exists():
            missing.append(path.name)
            continue

        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {path.name}: {e}")

    if missing:
        st.warning(
            "Missing model files: " + ", ".join(missing)
        )

    return models


def aligned_input(model, overrides=None):
    """Build one inference row using exactly the feature names expected by the model."""
    overrides = overrides or {}

    names = []
    for attr in ("feature_names_", "feature_name_"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                names = list(value)
            except Exception:
                names = []
            if names:
                break

    if not names:
        names = list(overrides.keys())

    row = {}
    for feature in names:
        if feature in overrides:
            value = overrides[feature]
        elif feature in df.columns:
            numeric = pd.to_numeric(df[feature], errors="coerce")
            value = float(numeric.median()) if numeric.notna().any() else 0.0
        else:
            value = 0.0

        try:
            if pd.isna(value) or not np.isfinite(float(value)):
                value = 0.0
        except Exception:
            value = 0.0

        row[feature] = float(value)

    return pd.DataFrame([row], columns=names).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def load_metrics():
    p = MODEL_DIR/"metrics.json"
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: pass
    return {}

try:
    if uploaded_file is not None:
        df = prepare(load_uploaded_data(uploaded_file))
        DATA_SOURCE = f"Uploaded: {uploaded_file.name}"
    else:
        df = prepare(load_data())
        DATA_SOURCE = "Google Drive"
except Exception as e:
    df = pd.DataFrame()
    DATA_SOURCE = "Unavailable"
    st.error(f"Data loading failed: {e}")

models = load_models()
metrics = load_metrics()

# ========================= SIDEBAR ==========================
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">☀</div>
      <div>
        <div class="brand-title">SOLAR AI</div>
        <div class="brand-sub">INTELLIGENCE COMMAND CENTER</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Control Panel</div>', unsafe_allow_html=True)

    pages = {
        "◈  Command Center":"Command Center",
        "⚡  Predictive Power":"Predictive Power",
        "◉  Fault Intelligence":"Fault Intelligence",
        "⌁  Root Cause":"Root Cause",
        "▣  Severity Monitor":"Severity Monitor",
        "◇  Model Laboratory":"Model Laboratory",
        "▤  Data Observatory":"Data Observatory",
        "⌘  Explainability":"Explainability",
        "⚙  System Settings":"System Settings",
    }
    page_label = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    page = pages[page_label]

    st.markdown('<div class="section-label">Appearance</div>', unsafe_allow_html=True)
    theme = st.toggle("Dark mode", value=st.session_state.dark)
    if theme != st.session_state.dark:
        st.session_state.dark = theme
        st.rerun()

    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
    if not df.empty:
        st.success("DATA  •  ONLINE")
    else:
        st.warning("DATA  •  WAITING")
    st.caption(f"Models loaded: {len(models)}/8")
    st.caption("Pipeline: CatBoost + LightGBM")

# ========================= TOP BAR ==========================
st.markdown(
    '<div style="display:flex;justify-content:space-between;align-items:center;">'
    '<div><span class="small-muted">SOLAR FARM / AI OPERATIONS</span></div>'
    '<div class="status">● SYSTEM READY</div></div>',
    unsafe_allow_html=True
)

# ========================= COMMAND CENTER ===================
if page == "Command Center":
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">AI-POWERED SOLAR OPERATIONS</div>
      <div class="hero-title">Solar AI Command Center</div>
      <div class="hero-text">
        A production-style control surface for solar power forecasting, fault detection,
        root-cause analysis, severity assessment and model explainability.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("Upload a CSV from the sidebar or use the default Google Drive dataset.")
    else:
        total = len(df)
        devices = df["device"].nunique() if "device" in df else 0
        fault_rate = df["is_faulted"].mean()*100 if "is_faulted" in df else 0
        avg_power = df["active_power"].mean() if "active_power" in df else 0
        avg_pr = df["performance_ratio"].mean() if "performance_ratio" in df else 0

        c = st.columns(5)
        for col, args in zip(c, [
            ("OBSERVATIONS",f"{total:,}","dataset records"),
            ("ASSETS",f"{devices:,}","monitored devices"),
            ("FAULT RATE",f"{fault_rate:.2f}%","overall exposure"),
            ("AVG POWER",f"{avg_power:.2f}","active power"),
            ("AVG PR",f"{avg_pr:.2%}","performance ratio"),
        ]):
            with col: kpi(*args)

        st.write("")
        left,right = st.columns([1.35,1])
        with left:
            card("Power Performance", "Recent active-power behavior")
            if "time" in df and "active_power" in df:
                t = df.dropna(subset=["time"]).tail(6000)
                chart(px.area(t, x="time", y="active_power", labels={"active_power":"Power"}), 340)

        with right:
            card("System Health", "Healthy vs faulted observations")
            if "is_faulted" in df:
                s = df["is_faulted"].map({0:"Healthy",1:"Faulted"}).value_counts()
                fig = px.pie(values=s.values,names=s.index,hole=.68)
                chart(fig,340)

        a,b,c = st.columns(3)
        with a:
            card("Fault Exposure by Device")
            if "device" in df and "is_faulted" in df:
                x=df.groupby("device")["is_faulted"].mean().mul(100).sort_values(ascending=False).head(10)
                chart(
    px.bar(
        x=x.index.tolist(),
        y=x.values.tolist(),
        labels={"x":"Device","y":"Fault %"}
    ),
    300
)
        with b:
            card("Fault Type Mix")
            if "fault_labels" in df:
                s=df["fault_labels"].value_counts().head(8)
                chart(px.pie(values=s.values,names=s.index,hole=.55),300)
        with c:
            card("Severity Profile")
            if "fault_severity" in df:
                severity_df = (
                    df["fault_severity"]
                    .map({
                        0: "None",
                        1: "Low",
                        2: "Medium",
                        3: "High"
                    })
                    .value_counts()
                    .rename_axis("Severity")
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    severity_df,
                    x="Severity",
                    y="Count",
                    labels={
                        "Severity": "Severity",
                        "Count": "Count"
                    }
                )

                chart(fig, 300)

# ========================= POWER =============================
elif page == "Predictive Power":
    st.title("⚡ Predictive Power")
    st.caption("Interactive active-power forecasting using the same feature logic as the project notebook.")

    features = [
        "clear_sky_irradiance","ambient_temp","module_temp","inverter_temp",
        "cloud_cover","sun_elevation","sun_azimuth","day_length_hours",
        "hour","month","week","day_of_week","is_daylight"
    ]
    features = [f for f in features if f in df.columns]
    if not features:
        st.warning("Dataset is required.")
    else:
        c1,c2,c3 = st.columns(3)
        vals={}
        for i,f in enumerate(features):
            with [c1,c2,c3][i%3]:
                med=float(df[f].median())
                vals[f]=st.number_input(f.replace("_"," ").title(), value=med, key="p_"+f)

        st.divider()
        alg = st.radio("Inference engine",["CatBoost","LightGBM"],horizontal=True)
        model = models.get(f"Power / {alg}")

        if st.button("RUN POWER FORECAST", type="primary", use_container_width=True):
            if model:
                X = aligned_input(model, vals)
                pred = float(model.predict(X)[0])
                c1,c2=st.columns([1,1.5])
                with c1:
                    kpi("PREDICTED ACTIVE POWER",f"{pred:.3f}","model output")
                with c2:
                    fig=go.Figure(go.Indicator(
                        mode="gauge+number", value=max(pred,0),
                        title={"text":"Predicted Power"},
                        gauge={"axis":{"range":[0,max(float(df.active_power.max())*1.1,1)]}}
                    ))
                    chart(fig,300)
            else:
                st.warning("Model files are not available. Check the `models/` folder in the GitHub repository.")

# ========================= FAULT =============================
elif page == "Fault Intelligence":
    st.title("◉ Fault Intelligence")
    st.caption("Binary health classification with probability and operational risk.")

    features=[
        "active_power","irradiance","clear_sky_irradiance","ambient_temp","module_temp",
        "inverter_temp","dc_voltage","dc_current","ac_voltage","ac_current",
        "performance_ratio","cloud_cover","sun_elevation","sun_azimuth",
        "day_length_hours","hour","month","week","day_of_week","is_daylight",
        "dc_power","ac_power","inverter_efficiency"
    ]
    features=[f for f in features if f in df.columns]
    if features:
        vals={}
        cols=st.columns(3)
        for i,f in enumerate(features):
            with cols[i%3]:
                vals[f]=st.number_input(f.replace("_"," ").title(), value=float(df[f].median()), key="f_"+f)
        alg=st.radio("Classifier",["CatBoost","LightGBM"],horizontal=True)
        if st.button("ANALYZE SYSTEM HEALTH",type="primary",use_container_width=True):
            model=models.get(f"Fault / {alg}")
            if model:
                X = aligned_input(model, vals)
                pred = int(np.asarray(model.predict(X)).ravel()[0])
                prob=float(model.predict_proba(X)[0][1])
                c1,c2=st.columns([1,1.3])
                with c1:
                    kpi("SYSTEM STATUS","FAULT" if pred else "HEALTHY",f"{prob:.1%} fault probability")
                with c2:
                    fig=go.Figure(go.Indicator(
                        mode="gauge+number", value=prob*100,
                        title={"text":"Fault Probability (%)"},
                        gauge={"axis":{"range":[0,100]}}
                    ))
                    chart(fig,280)
            else:
                st.warning("Model files are not available. Check the `models/` folder in the GitHub repository.")

# ========================= ROOT CAUSE ========================
elif page == "Root Cause":
    st.title("⌁ Root Cause Intelligence")
    st.caption("Multi-class fault component diagnosis.")
    if "fault_labels" in df:
        s=df["fault_labels"].value_counts()
        left,right=st.columns([1.15,1])
        with left:
            card("Historical Root-Cause Distribution")
            chart(
    px.bar(
        x=s.index.tolist(),
        y=s.values.tolist(),
        labels={"x":"Fault label","y":"Occurrences"}
    ),
    400
)
        with right:
            card("Cause Mix")
            chart(px.pie(values=s.values,names=s.index,hole=.58),400)

        st.info("For live diagnosis, the Model Laboratory uses the trained Model 3 classifier and returns class probabilities.")
    else:
        st.info("Load the project dataset.")

# ========================= SEVERITY ==========================
elif page == "Severity Monitor":
    st.title("▣ Severity Monitor")
    st.caption("Operational view of fault severity and exposure.")
    if "fault_severity" in df:
        sev=df["fault_severity"].map({0:"None",1:"Low",2:"Medium",3:"High"})
        counts=sev.value_counts().reindex(["None","Low","Medium","High"]).fillna(0)
        c=st.columns(4)
        for col,label in zip(c,counts.index):
            with col: kpi(label,f"{int(counts[label]):,}","observations")
        st.write("")
        left,right=st.columns(2)
        with left:
            card("Severity Distribution")
            chart(
    px.bar(
        x=counts.index.tolist(),
        y=counts.values.tolist()
    ),
    350
)
        with right:
            card("Severity Exposure")
            chart(px.pie(values=counts.values,names=counts.index,hole=.62),350)

# ========================= MODEL LAB =========================
elif page == "Model Laboratory":
    st.title("◇ Model Laboratory")
    st.caption("A formal evaluation center for the eight trained CatBoost / LightGBM estimators.")

    if metrics:
        rows=[]
        for task,data in metrics.items():
            if isinstance(data,dict):
                for model_name,vals in data.items():
                    row={"Task":task,"Model":model_name,**vals}
                    rows.append(row)
        if rows:
            mdf=pd.DataFrame(rows)
            st.dataframe(mdf,use_container_width=True,hide_index=True)
            numeric=[x for x in mdf.columns if x not in ["Task","Model"]]
            if numeric:
                metric=numeric[0]
                fig=px.bar(mdf,x="Task",y=metric,color="Model",barmode="group")
                chart(fig,420)
    else:
        st.info("No saved evaluation metrics were found in `models/metrics.json`.")

    st.divider()
    st.markdown("### Model Registry")
    registry = [
        ("M1","Active Power","Regression","MAE / RMSE / R²"),
        ("M2","Fault Detection","Binary Classification","Accuracy / F1 / ROC-AUC"),
        ("M3","Fault Component","Multiclass Classification","Macro F1 / Weighted F1"),
        ("M4","Fault Severity","Multiclass Classification","Macro F1 / Weighted F1"),
    ]
    st.dataframe(pd.DataFrame(registry,columns=["ID","Task","Type","Primary Metrics"]),use_container_width=True,hide_index=True)

# ========================= DATA ==============================
elif page == "Data Observatory":
    st.title("▤ Data Observatory")
    st.caption("Explore the dataset behind the intelligence layer.")
    if df.empty:
        st.info("Upload a CSV from the sidebar or use the default Google Drive dataset.")
    else:
        n1,n2,n3=st.columns(3)
        with n1:kpi("Rows",f"{len(df):,}")
        with n2:kpi("Columns",f"{df.shape[1]:,}")
        with n3:kpi("Missing Values",f"{int(df.isna().sum().sum()):,}")
        tab1,tab2,tab3=st.tabs(["Distribution","Correlation","Raw Data"])
        with tab1:
            numeric=df.select_dtypes(include=np.number).columns.tolist()
            if numeric:
                col=st.selectbox("Feature",numeric)
                chart(px.histogram(df,x=col,nbins=60,marginal="box"),380)
        with tab2:
            numeric=df.select_dtypes(include=np.number)
            if not numeric.empty:
                chart(px.imshow(numeric.corr(),text_auto=".2f",aspect="auto"),650)
        with tab3:
            st.dataframe(df.head(1000),use_container_width=True,height=500)

# ========================= EXPLAINABILITY ====================
elif page == "Explainability":
    st.title("⌘ Explainability")
    st.caption("Understand which variables drive the trained models.")
    if not models:
        st.info("No model artifacts are currently available in the `models/` folder.")
    else:
        choice=st.selectbox("Model",list(models.keys()))
        model=models[choice]
        if hasattr(model,"feature_importances_"):
            imp=np.asarray(model.feature_importances_)
            names=getattr(model,"feature_name_",None)
            if names is None:
                names=[f"Feature {i+1}" for i in range(len(imp))]
            x=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False).head(15)
            chart(px.bar(x.sort_values("Importance"),x="Importance",y="Feature",orientation="h"),500)
        elif hasattr(model,"get_feature_importance"):
            imp=np.asarray(model.get_feature_importance())
            names=list(getattr(model,"feature_names_",[]))
            if len(names)!=len(imp): names=[f"Feature {i+1}" for i in range(len(imp))]
            x=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False).head(15)
            chart(px.bar(x.sort_values("Importance"),x="Importance",y="Feature",orientation="h"),500)

# ========================= SETTINGS ==========================
elif page == "System Settings":
    st.title("⚙ System Settings")
    st.caption("Formal system control and deployment information.")
    a,b=st.columns(2)
    with a:
        card("Data Source")
        st.write(DATA_SOURCE)
        card("Model Registry")
        st.write(f"{len(models)} / 8 model artifacts available")
    with b:
        card("Theme")
        st.write("Dark" if DARK else "Light")
        card("Architecture")
        st.write("Streamlit → Feature Layer → CatBoost / LightGBM → Analytics")
    st.divider()
    st.download_button(
        "DOWNLOAD PROJECT METADATA",
        json.dumps({"models_loaded":list(models.keys()),"data_source":DATA_SOURCE},indent=2),
        file_name="solar_ai_system_metadata.json",
        mime="application/json",
        use_container_width=True
    )

st.divider()
st.caption("SOLAR AI COMMAND CENTER  •  ML CAPSTONE  •  CATBOOST + LIGHTGBM")
