# Retail Banking Fraud Detection - Capstone

**Domain:** Finance (card-not-present / e-commerce)
**Goal:** Reduce fraud losses ≥20% while keeping FPR ≤ 1% and staying within alert capacity.
**Dataset:** ULB Credit Card Fraud (Kaggle/ULB) — highly imbalanced.

## Results (TEST set, chosen policy FPR≈0.5%)
- Model: RandomForest (17 selected features)
- Threshold: chosen on validation to satisfy FPR ≤ 0.5%, then applied to TEST
- Alerts: 100
- Precision: 16.0%
- Recall: 72.7%
- FPR: 0.3%
- Expected Value (₱): 99,400
- Confusion (TP/FP/FN/TN): 16 / 84 / 6 / 28,375

> We also reported validation comparisons across Logistic, RandomForest, XGBoost with EV at K∈{100,250,500,1000} and FPR caps {0.5%,1%,2%}.

## Business framing (Step 1)
- Task: anomaly scoring + supervised fraud classification benchmark
- KPIs: fraud loss saved; Precision@K (K=500/day capacity); FPR ≤ 1%
- Tech metrics: PR-AUC, ROC AUC, Precision@K, Recall@K
- Decision policy: choose threshold τ to meet constraints and maximize Expected Value (EV)
- Costs used: C_fraud=₱7,500; C_review=₱80; C_fp=₱150

## Data (Step 2)
- ULB dataset; no missing values; all numeric.
- Time = seconds since first transaction; Amount = EUR; Class = 0/1.
- V1..V28 anonymized PCA-like components.
- Data dictionary: see reports/data_dictionary.csv.

## EDA & Features (Step 3)
- Engineered: Amount_log, robust-z Amount_rz, time-of-day Time_sin/Time_cos.
- PCA visualization (variance concentrated + sparse outliers).
- Feature selection: elastic-net → 17/33 kept.

## Modeling & Thresholds (Step 4)
- Models: Logistic, RandomForest, XGBoost (class-weighted / scale_pos_weight).
- Validation grid: EV at top-K and FPR caps; pick τ from VAL then report TEST.
- Artifacts: reports/model_compare_val.csv, reports/test_operating_point.csv.

## Explainability & Fairness (Step 5)
- SHAP global importance (beeswarm) + local waterfalls (TP/FP).
- Fairness slices (proxy): Amount quartiles & time-of-day.
  - Lower precision for ₱3.8–15.1 band → propose per-segment thresholds and monthly monitoring.
- Artifacts: reports/figures/shap_*.png, reports/fairness_slices_test.csv.

## How to run
1) Create environment:
    pip install -r requirements.txt
2) Open notebooks in notebooks/ (or run the Colab cells in order):
    01_eda_preprocess.ipynb
    02_models_thresholds.ipynb
    03_explain_fairness.ipynb
3) Place the dataset at data/creditcard.csv if you want to re-run from scratch.

## Repo layout
fraud-capstone/
  README.md
  requirements.txt
  .gitignore
  notebooks/
  src/
  data/               (empty; do not commit raw data)
  models/             (optional saved model(s))
  reports/
    data_dictionary.csv
    model_compare_val.csv
    test_operating_point.csv
    figures/
      ev_vs_k.png
      shap_beeswarm.png
      shap_waterfall_TP.png
      shap_waterfall_FP.png

## Credits
- Dataset: “Credit Card Fraud Detection” — ULB Machine Learning Group (via Kaggle).
- Author: <Your Name>
- License: MIT (repo code only). Dataset subject to original license.