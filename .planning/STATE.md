# StreamShield State

- **Completed Phases**:
  - Phase 01: Synthetic Data Generation (`src/generator.py` $\to$ `data/round1.csv`, `data/round2.csv`)
  - Phase 02: Heuristic Rules Baseline (`src/baseline.py`, PR-AUC: 0.1933, Recall: 19.27%)
  - Phase 03: Stateful Feature Engine (`src/features.py` $\to$ `data/features_round1.csv`, `data/features_round2.csv`)
- **Current Phase**: Phase 04 — Hybrid ML Model & Calibration
- **Phase Status**: Ready for implementation (`src/train.py`)
- **Target File**: `src/train.py`
- **Output Artifacts Expected**:
  - `data/model_lgb.joblib`
  - `data/model_calibrator.joblib`
  - `data/model_iforest.joblib`
  - Comparative benchmark table: Rules vs LightGBM vs Hybrid Blend
- **Environment**:
  - Python 3.11.17 (`venv`)
  - Virtual Environment: `venv` active
  - Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `networkx`, `matplotlib`, `streamlit`, `plotly`

