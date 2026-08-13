
"""
Train and serialize the exact four ML tasks from Explained_Final_Project.ipynb.

Run:
    python train_models.py

Expected dataset:
    data/Synthetic-Solar-Farm-Stream-No-Repair.csv
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

BASE=Path(__file__).parent
DATA=BASE/"data"/"Synthetic-Solar-Farm-Stream-No-Repair.csv"
MODELS=BASE/"models"
MODELS.mkdir(exist_ok=True)

df=pd.read_csv(DATA)
df["downtime_start_time"]=pd.to_datetime(df["downtime_start_time"], errors="coerce")
df["time"]=pd.to_datetime(df["time"], errors="coerce")

# Remove rows where the main timestamp cannot be parsed.
df=df.dropna(subset=["time"]).sort_values("time")
df["hour"]=df["time"].dt.hour
df["month"]=df["time"].dt.month
df["day_of_week"]=df["time"].dt.dayofweek
df["is_daylight"]=df["hour"].between(6,18).astype(int)
df=df.set_index("time")

df["dc_power"]=df["dc_voltage"]*df["dc_current"]
df["ac_power"]=df["ac_voltage"]*df["ac_current"]
df["inverter_efficiency"]=np.where(df["dc_power"]>0,df["ac_power"]/df["dc_power"],0.0).clip(0,1)
df["fault_severity"]=df["fault_severity"].map({"none":0,"low":1,"medium":2,"high":3})

# Make sure the engineered numeric columns are numeric.
for col in ["hour","month","day_of_week","is_daylight","dc_power","ac_power","inverter_efficiency","week"]:
    if col in df.columns:
        df[col]=pd.to_numeric(df[col], errors="coerce")

def split_group(g, frac=.8):
    cut=int(np.floor(len(g)*frac))
    return g.iloc[:cut],g.iloc[cut:]

df["week"]=df.index.isocalendar().week
trs,tes=[],[]
for _,g in df.groupby(["device","week"],sort=False):
    tr,te=split_group(g)
    trs.append(tr); tes.append(te)
train=pd.concat(trs).sort_index()
test=pd.concat(tes).sort_index()

leakage=[
    "device","is_faulted","fault_labels","fault_severity",
    "fault_soiling","fault_inverter_overheat","fault_tracker_stuck",
    "fault_dc_string_outage","downtime_duration_min",
    # Raw datetime cannot be passed to CatBoost/LightGBM as a numeric feature.
    "downtime_start_time"
]
m1_extra=[
    "active_power","ac_power","dc_power","ac_voltage","ac_current",
    "dc_voltage","dc_current","inverter_efficiency",
    "performance_ratio","irradiance"
]
m1_features=[c for c in df.columns if c not in leakage+m1_extra]
m_features=[c for c in df.columns if c not in leakage]

# Final safety check: never send datetime/object columns such as NaT to the models.
datetime_features=[c for c in m_features if pd.api.types.is_datetime64_any_dtype(df[c])]
if datetime_features:
    m_features=[c for c in m_features if c not in datetime_features]
m1_features=[c for c in m1_features if not pd.api.types.is_datetime64_any_dtype(df[c])]

# Replace infinities with NaN. CatBoost/LightGBM can handle numeric NaN values.
train=train.replace([np.inf,-np.inf], np.nan)
test=test.replace([np.inf,-np.inf], np.nan)

Xtr1=train[m1_features]; ytr1=train["active_power"]
Xte1=test[m1_features]; yte1=test["active_power"]

Xtr2=train[m_features]; ytr2=train["is_faulted"]
Xte2=test[m_features]; yte2=test["is_faulted"]

Xtr3=train[m_features]; ytr3=train["fault_labels"]
Xte3=test[m_features]; yte3=test["fault_labels"]
masktr=ytr3!="dc_string_outage|downtime"
maskte=yte3!="dc_string_outage|downtime"
Xtr3,ytr3=Xtr3[masktr],ytr3[masktr]
Xte3,yte3=Xte3[maskte],yte3[maskte]

Xtr4=train[m_features]; ytr4=train["fault_severity"]
Xte4=test[m_features]; yte4=test["fault_severity"]

# M1
cat1=CatBoostRegressor(iterations=600,learning_rate=.05,depth=8,loss_function="RMSE",random_state=42,verbose=False)
lgb1=LGBMRegressor(n_estimators=600,learning_rate=.05,max_depth=8,random_state=42,verbose=-1)
cat1.fit(Xtr1,ytr1); lgb1.fit(Xtr1,ytr1)
p1c=cat1.predict(Xte1); p1l=lgb1.predict(Xte1)

# M2
classes=np.unique(ytr2); weights=dict(zip(classes,compute_class_weight("balanced",classes=classes,y=ytr2)))
cat2=CatBoostClassifier(iterations=500,learning_rate=.05,depth=8,loss_function="Logloss",eval_metric="F1",class_weights=weights,random_state=42,verbose=False)
lgb2=LGBMClassifier(n_estimators=500,learning_rate=.05,max_depth=8,class_weight=weights,random_state=42,verbose=-1)
cat2.fit(Xtr2,ytr2); lgb2.fit(Xtr2,ytr2)
p2c=cat2.predict(Xte2).ravel(); p2l=lgb2.predict(Xte2).ravel()

# M3
classes=np.unique(ytr3); weights=dict(zip(classes,compute_class_weight("balanced",classes=classes,y=ytr3)))
cat3=CatBoostClassifier(iterations=500,learning_rate=.05,depth=8,loss_function="MultiClass",eval_metric="TotalF1",class_weights=weights,random_state=42,verbose=False)
lgb3=LGBMClassifier(n_estimators=500,learning_rate=.05,max_depth=8,objective="multiclass",class_weight=weights,random_state=42,verbose=-1)
cat3.fit(Xtr3,ytr3); lgb3.fit(Xtr3,ytr3)
p3c=cat3.predict(Xte3).ravel(); p3l=lgb3.predict(Xte3).ravel()

# M4
classes=np.unique(ytr4); weights=dict(zip(classes,compute_class_weight("balanced",classes=classes,y=ytr4)))
cat4=CatBoostClassifier(iterations=500,learning_rate=.05,depth=8,loss_function="MultiClass",eval_metric="TotalF1",class_weights=weights,random_state=42,verbose=False)
lgb4=LGBMClassifier(n_estimators=500,learning_rate=.05,max_depth=8,objective="multiclass",class_weight=weights,random_state=42,verbose=-1)
cat4.fit(Xtr4,ytr4); lgb4.fit(Xtr4,ytr4)
p4c=cat4.predict(Xte4).ravel(); p4l=lgb4.predict(Xte4).ravel()

for name,obj in {
"cat_reg_m1":cat1,"lgbm_reg_m1":lgb1,
"cat_model_m2":cat2,"lgbm_model_m2":lgb2,
"cat_model_m3":cat3,"lgbm_model_m3":lgb3,
"cat_model_m4":cat4,"lgbm_model_m4":lgb4}.items():
    joblib.dump(obj,MODELS/(name+".joblib"))

metrics={
"Active Power":{
"CatBoost":{"MAE":mean_absolute_error(yte1,p1c),"RMSE":mean_squared_error(yte1,p1c)**.5,"R2":r2_score(yte1,p1c)},
"LightGBM":{"MAE":mean_absolute_error(yte1,p1l),"RMSE":mean_squared_error(yte1,p1l)**.5,"R2":r2_score(yte1,p1l)}
},
"Fault Detection":{
"CatBoost":{"Accuracy":accuracy_score(yte2,p2c),"F1 Macro":f1_score(yte2,p2c,average="macro"),"F1 Weighted":f1_score(yte2,p2c,average="weighted"),"ROC AUC":roc_auc_score(yte2,cat2.predict_proba(Xte2)[:,1])},
"LightGBM":{"Accuracy":accuracy_score(yte2,p2l),"F1 Macro":f1_score(yte2,p2l,average="macro"),"F1 Weighted":f1_score(yte2,p2l,average="weighted"),"ROC AUC":roc_auc_score(yte2,lgb2.predict_proba(Xte2)[:,1])}
},
"Fault Component":{
"CatBoost":{"Accuracy":accuracy_score(yte3,p3c),"F1 Macro":f1_score(yte3,p3c,average="macro"),"F1 Weighted":f1_score(yte3,p3c,average="weighted")},
"LightGBM":{"Accuracy":accuracy_score(yte3,p3l),"F1 Macro":f1_score(yte3,p3l,average="macro"),"F1 Weighted":f1_score(yte3,p3l,average="weighted")}
},
"Fault Severity":{
"CatBoost":{"Accuracy":accuracy_score(yte4,p4c),"F1 Macro":f1_score(yte4,p4c,average="macro"),"F1 Weighted":f1_score(yte4,p4c,average="weighted")},
"LightGBM":{"Accuracy":accuracy_score(yte4,p4l),"F1 Macro":f1_score(yte4,p4l,average="macro"),"F1 Weighted":f1_score(yte4,p4l,average="weighted")}
}
}
(BASE/"models"/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
(BASE/"models"/"feature_schema.json").write_text(json.dumps({"m1":m1_features,"m2":m_features,"m3":m_features,"m4":m_features},indent=2),encoding="utf-8")
print("Training complete. 8 model artifacts + metrics saved in models/.")
