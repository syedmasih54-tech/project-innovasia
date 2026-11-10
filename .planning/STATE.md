# StreamShield State

- **Completed Phases**:
  - Phase 01: Synthetic Data Generation (`src/generator.py` $\to$ `data/round1.csv`, `data/round2.csv`)
  - Phase 02: Heuristic Rules Baseline (`src/baseline.py`, PR-AUC: 0.1933, Recall: 19.27%)
- **Current Phase**: Phase 03 — Stateful Feature Engine
- **Phase Status**: Specified (`03-SPEC.md`) & Ready for implementation
- **Target File**: `src/features.py`
- **Output Artifacts Expected**:
  - `data/features_round1.csv`
  - `data/features_round2.csv`
- **Environment**:
  - Python 3.11.17 (`venv`)
  - Virtual Environment: `venv` active
  - Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `networkx`, `matplotlib`, `streamlit`, `plotly`

