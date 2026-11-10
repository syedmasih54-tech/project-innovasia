# StreamShield — Real-Time UPI Fraud & Mule Ring Detection Engine

## Overview
**Innovasia 2026 · PS-01 · Fintech Hackathon**

StreamShield is an ultra-low-latency fraud prevention engine built for India's Unified Payments Interface (UPI). Because UPI transfers settle instantly and are irreversible, StreamShield evaluates every incoming payment transaction in real time before funds move, rendering an automated three-way decision:
1. **ALLOW**: Safe payment, proceed instantly.
2. **STEP-UP**: Moderate risk; trigger additional customer verification (e.g. biometric / OTP friction).
3. **HOLD**: Critical risk; block transaction immediately.

StreamShield pairs high-throughput decisioning with real-time explainability (generating up to 3 plain-English reasons per flagged transaction) and graph-based mule ring detection.

---

## Performance Targets
- **Decision Latency**: $\le 150\text{ ms}$ at **p99** on a single laptop/workstation.
- **Throughput**: **2,000 Transactions Per Second (TPS)** under live streaming simulation.
- **Fraud Distribution**: $< 0.1\%$ fraud rate (severe class imbalance).
- **Cold-Start Resilience**: ~15% brand-new accounts with zero transaction history.
- **Evaluation Metrics**: Precision-Recall AUC (PR-AUC) and Recall at 1% False Positive Rate (FPR), rather than deceptive overall accuracy.

---

## Architecture & Technology Stack
- **Language**: Python 3.11
- **Feature Store & Memory**: Pure Python stateful in-memory structures (`deque`, `dict`) optimized for sub-millisecond updates without heavy ORM or database latency.
- **Detection Ensemble**:
  - Supervised Gradient Boosting: `LightGBM` (with `scale_pos_weight` handling rare fraud).
  - Unsupervised Anomaly Detection: `IsolationForest` (for novel, zero-day patterns).
  - Calibration: `IsotonicRegression` ensuring risk scores correspond to empirical fraud probabilities.
- **Decision Engine**: Cost-utility expected loss minimization balancing stolen money recovery vs. customer checkout friction abandonment.
- **Explainability**: TreeSHAP feature contribution values mapped to human-readable sentence templates.
- **Graph Clustering**: Disjoint-set union / `networkx` connected components identifying coordinated mule rings.
- **Operational Dashboard**: `Streamlit` + `Plotly` live monitor showing incoming telemetry, network graph rings, and latency histograms.
- **Drift Adaptation**: Efficient recalibration on Round 2 adversarial drift without full offline retraining.

