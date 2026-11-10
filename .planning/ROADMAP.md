# StreamShield Roadmap

**Milestone 1: Production Hackathon Prototype (Innovasia 2026 PS-01)**

| Phase | Title | Guide Step | Deliverable | Status |
|:---:|:---|:---:|:---|:---:|
| **01** | Synthetic Data Generation | Step 1 | `src/generator.py` generating `data/round1.csv` & `data/round2.csv` | **Completed** |
| **02** | Heuristic Rules Baseline | Step 2 | `src/baseline.py` with temporal train/test split & PR-AUC evaluation | **Completed** |
| **03** | Stateful Feature Engine | Step 3 | `src/features.py` (O(1) memory, pass-through ratio, account age) | **Completed** |
| **04** | Hybrid ML Model & Calibration | Step 4 | `src/train.py` (LightGBM + Isolation Forest + Isotonic Calibration) | **Completed** |
| **05** | Cost-Utility Decision & Explanations | Step 5 | `src/decision.py` (Allow/Step-up/Hold + 3 plain-text reasons) | **Completed** |
| **06** | 2,000 TPS Streaming Replay & Latency Benchmark | Step 6 | `src/replay.py` (`decisions.jsonl`, `metrics.json`, `latency.png`) | **Completed** |
| **07** | Graph Mule Ring Detection | Step 7 | `src/rings.py` (`data/rings.json` + ring detection recall & time) | **Completed** |
| **08** | Real-Time Operations Dashboard | Step 8 | `src/make_sample_outputs.py` & `app/dashboard.py` (Streamlit + Plotly) | **Completed** |
| **09** | Round 2 Concept Drift Recalibration | Step 9 | `src/round2.py` (lightweight adaptation without full retraining) | **Completed** |
| **10** | End-to-End Orchestrator & Documentation | Step 10 | `run_all.py`, `README.md`, `requirements.txt` | **Completed** |
| **11** | Demo Presentation & Final Verification | Step 11 | Verified metrics & slide-ready benchmark table | **Completed** |

