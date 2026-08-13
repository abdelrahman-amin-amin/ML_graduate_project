# ☀️ Solar AI Command Center — PRO

A professional Streamlit application built around the Solar ML capstone notebook.

## What makes it stronger than the notebook?

The notebook is for experimentation and training. This application is the **product layer**:
- formal control panel and navigation
- Dark / Light website-style theme switch
- command-center KPIs
- power forecasting
- fault probability + risk gauge
- root-cause analytics
- severity monitoring
- model laboratory
- explainability
- data observatory
- saved model registry
- downloadable system metadata

## Run locally

1. Put `Synthetic-Solar-Farm-Stream-No-Repair.csv` in `data/`.
2. Install:

```bash
pip install -r requirements.txt
```

3. Train and serialize the 8 models:

```bash
python train_models.py
```

4. Launch:

```bash
streamlit run streamlit_app.py
```

## Important

The training script follows the feature engineering and four model tasks in `Explained_Final_Project.ipynb`.

The app can be deployed to Streamlit Community Cloud after the model artifacts are prepared. For a lightweight cloud deployment, consider storing the model files with Git LFS or an external model store if their size is large.
