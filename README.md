# ML_graduate_project

# ☀️ Solar Farm Fault Detection

Machine Learning pipeline for predictive maintenance of photovoltaic (PV) solar farms — forecasting expected power output, detecting faults, classifying fault type, and estimating fault severity from inverter telemetry.

## 📌 Project Overview

Photovoltaic assets often lose economic return silently: a soiled panel or an overheating inverter keeps producing power, just at gradually declining efficiency. Manual inspection doesn't scale to large device fleets, and static-threshold alarms can't distinguish an expected production drop (night, clouds) from a genuine fault.

This project closes that gap with an **integrated four-model pipeline** that learns the relationship between environmental conditions and electrical output, then uses it to forecast expected power, detect faults, classify the fault type, and estimate severity — moving from reactive/scheduled maintenance to condition-based maintenance.

## 🎯 Objectives

- Build a regression model to forecast `active_power` from environmental/temporal features
- Build a binary classifier to detect normal vs. faulted operation
- Build a multi-class classifier to identify the fault type (soiling, inverter overheating, tracker stuck, DC string outage)
- Build a model to estimate fault severity (none / low / medium / high)
- Compare **CatBoost** and **LightGBM** on each task, and combine them via ensemble blending

## 📊 Dataset

- **Source:** `Synthetic-Solar-Farm-Stream-No-Repair.csv` — synthetic telemetry simulating a real solar farm
- **Size:** 525,600 rows × 25 columns (1 full year, 5-minute resolution, 2025)
- **Devices:** 5 inverters (PV000–PV004)
- **Feature groups:** electrical, environmental, thermal, fault indicators, metadata
- **Target variables:**

  | Target | Type | Model |
  |---|---|---|
  | `active_power` | continuous | 1 — Power forecasting |
  | `is_faulted` | binary | 2 — Fault detection |
  | `fault_labels` | multi-class | 3 — Fault type |
  | `fault_severity` | ordinal (0–3) | 4 — Fault severity |

Class imbalance is significant: faults represent 20.4% of readings overall, and each device is almost exclusively tied to one fault type — which required excluding the device identifier from training features to prevent leakage.

**Preprocessing highlights:**
- No missing values, no duplicated rows across all 525,600 × 25 cells
- Outliers (IQR method) in `irradiance`, `module_temp`, `inverter_temp`, `cloud_cover` retained as physically plausible, handled via class weighting
- Engineered features: `dc_power`, `ac_power`, `inverter_efficiency`, time features (`hour`, `month`, `day_of_week`, `is_daylight`), `downtime_duration_min`
- **Blocked time-based split by (device × week)** — 80/20 chronological split per block — instead of a single global time cut, since the monthly fault rate ranges from 0% to 41%
- Leakage-aware, model-specific feature selection (e.g. excluding device ID from Models 2–4)

## ⚙️ Technologies Used

| Library | Purpose |
|---|---|
| pandas / numpy | Data loading & processing |
| matplotlib / seaborn | Visualization & EDA |
| scikit-learn | Metrics, splitting, class weighting |
| catboost | CatBoostRegressor / CatBoostClassifier |
| lightgbm | LGBMRegressor / LGBMClassifier |
| optuna | Automated hyperparameter tuning |
| streamlit | Interactive demo app |

**Requirements:** Python 3.11+

## 🤖 Machine Learning Models

```
Raw Telemetry CSV (525,600 × 25)
        ↓
Data Quality Audit (missing values / duplicates / outliers)
        ↓
Feature Engineering (power, efficiency, time features)
        ↓
Blocked Train/Test Split (device × week, 80/20)
        ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Model 1    │  Model 2    │  Model 3    │  Model 4    │
│  Power      │  Fault      │  Fault      │  Fault      │
│  Forecast   │  Detection  │  Type       │  Severity   │
│ (Regression)│  (Binary)   │(Multi-class)│  (Ordinal)  │
└─────────────┴─────────────┴─────────────┴─────────────┘
        ↓
CatBoost + LightGBM → Optuna Tuning → Class Weighting →
Ensemble Blend → Evaluation & Feature Importance
```

- **Algorithms:** CatBoost and LightGBM per task, plus an ensemble blend (probability averaging for classification, weighted averaging for regression)
- **Hyperparameter tuning:** automated Bayesian-style search via **Optuna** across trees/iterations (300–800), learning rate (0.01–0.3, log scale), tree depth (4–10), plus algorithm-specific parameters
- **Class imbalance handling:** balanced class weights (no resampling of raw data)
- **Evaluation metrics:** MAE / RMSE / R² for regression; Accuracy, Macro F1, Weighted F1, confusion matrix for classification; feature importance as an additional leakage check

## 📈 Results

| Model | Best Result |
|---|---|
| 1 — Power forecasting | **R² = 0.9971** (weighted blend) |
| 2 — Fault detection | **98.91% accuracy**, 98.92% weighted F1 (blend) |
| 3 — Fault-type classification | **98.44% accuracy**, 0.8098 Macro F1 (LightGBM) |
| 4 — Fault severity estimation | **98.73% accuracy** (blend), 0.9437 Macro F1 (LightGBM) |

Across the three classification tasks, LightGBM was consistently the strongest single model, and the ensemble blend gave a further tangible improvement in fault detection and severity estimation. Feature importance consistently highlighted thermal features (`module_temp`, `ambient_temp`, `inverter_temp`) as top fault indicators, aligning with the physical understanding of soiling and inverter overheating as common fault causes — supporting that the models learned genuine fault signal rather than leakage.


## 📂 Project Structure

```
.
├── Final_Project.ipynb   # Part 1: data loading, EDA, feature engineering, split
│                          # Part 2: model training, tuning, blending, evaluation
├── app.py                
├── models/                 
├── Synthetic-Solar-Farm-Stream-No-Repair.csv
├── requirements.txt
└── README.md
```

## 👥 Team Members

- Abdullah Mohamed Ibrahim
- Mohamed Ahmed Mohamed
- Suhaila Maher Shabat Al-Hamd
- Abdelrahman Amin Amin
- Mennah Ayman Ahmed
