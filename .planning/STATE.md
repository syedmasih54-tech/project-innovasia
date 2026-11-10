# StreamShield State

- **Completed Phases**:
  - Phase 01: Synthetic Data Generation (`src/generator.py` $\to$ `data/round1.csv`, `data/round2.csv`)
  - Phase 02: Heuristic Rules Baseline (`src/baseline.py`, PR-AUC: 0.1933, Recall: 19.27%)
  - Phase 03: Stateful Feature Engine (`src/features.py` $\to$ `data/features_round1.csv`, `data/features_round2.csv`)
  - Phase 04: Hybrid ML Model (`src/train.py`, PR-AUC: 0.8966, Recall @ 1% FPR: 98.33%)
  - Phase 05: Cost-Utility Decision & Explanations (`src/decision.py`, Rs 12.3 Lakhs saved, 9.2x ROI)
- **Current Phase**: Phase 06 — 2,000 TPS Streaming Replay & Latency Benchmark
- **Phase Status**: Ready for implementation (`src/replay.py`)
- **Target File**: `src/replay.py`
- **Output Artifacts Expected**:
  - `data/decisions.jsonl`
  - `data/metrics.json`
  - `data/latency.png`
  - Measured p50, p95, p99, and achieved TPS throughput
- **Environment**:
  - Python 3.11.17 (`venv`)
  - Virtual Environment: `venv` active
  - Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `networkx`, `matplotlib`, `streamlit`, `plotly`

