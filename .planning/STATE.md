# StreamShield State

- **Completed Phases**:
  - Phase 01: Synthetic Data Generation (`src/generator.py` $\to$ `data/round1.csv`, `data/round2.csv`)
  - Phase 02: Heuristic Rules Baseline (`src/baseline.py`, PR-AUC: 0.1933, Recall: 19.27%)
  - Phase 03: Stateful Feature Engine (`src/features.py` $\to$ `data/features_round1.csv`, `data/features_round2.csv`)
  - Phase 04: Hybrid ML Model (`src/train.py`, PR-AUC: 0.8966, Recall @ 1% FPR: 98.33%)
- **Current Phase**: Phase 05 — Cost-Utility Decision & Explanations
- **Phase Status**: Ready for implementation (`src/decision.py`)
- **Target File**: `src/decision.py`
- **Output Artifacts Expected**:
  - Rupee cost totals: Fraud Prevented vs Friction Caused
  - Decision explainability (up to 3 human-readable sentences per flagged transaction)
- **Environment**:
  - Python 3.11.17 (`venv`)
  - Virtual Environment: `venv` active
  - Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `networkx`, `matplotlib`, `streamlit`, `plotly`

