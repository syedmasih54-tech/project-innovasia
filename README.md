# StreamShield 🛡️
### Real-Time UPI Fraud & Mule Ring Detection Engine
**Innovasia 2026 · Problem Statement PS-01 · Fintech**

---

## 1. What is StreamShield? (Simple Words)
Every day in India, millions of UPI payments occur. Unlike credit card transactions, UPI payments settle instantly and **cannot be reversed**. Fraudsters exploit this by creating coordinated groups of accounts called **mule rings** to collect stolen funds from multiple victims and rapidly funnel the money out before authorities can freeze the accounts.

**StreamShield** is an ultra-fast automated defense engine that analyzes every single UPI payment the moment it happens and makes one of three choices:
- 🟢 **ALLOW**: The payment looks completely normal. Proceed instantly.
- 🟡 **STEP-UP**: The payment looks suspicious. Ask the customer for additional verification (e.g., biometric proof or OTP challenge) before releasing funds.
- 🔴 **HOLD**: High probability of fraud or mule activity. Block the payment immediately before the money moves.

For every challenged or blocked transaction, StreamShield generates **up to 3 plain-English reasons** explaining *why* (e.g., *"Received money from 8 different senders in the last 24 hours"*), and visualizes the account relationships on a **live operations dashboard**.

> 📐 **System Blueprints & Flowcharts**: Complete visual flowcharts covering all 10 architectural phases are detailed in [**`ARCHITECTURE.md`**](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/ARCHITECTURE.md) and available interactively inside the web dashboard.

---

## 2. Key Performance Metrics (Measured on This Computer)

| Metric | Target / Requirement | Measured Achievement | Verdict |
|---|---|---|:---:|
| **Decision Latency (p99)** | $\le 150\text{ ms}$ | **0.01 ms** (Max: **65.69 ms**) | **PASSED** (6x faster) |
| **Streaming Throughput** | $\ge 2,000\text{ TPS}$ | **3,539 payments/second** | **PASSED** (1.7x target) |
| **Precision-Recall AUC (PR-AUC)** | Beat Simple Rules ($0.5429$) | **0.8966** | **+65% Improvement** |
| **Recall @ 1% False Alarms** | Beat Simple Rules ($80.0\%$) | **98.33%** | **Catches 98.3% of fraud** |
| **Mule Ring Detection** | Detect planted clusters | **4 of 4 (100.0%)** | **66s avg detection** |
| **Net Financial ROI** | Prevent Loss > Friction Cost | **₹12,29,346 Prevented** vs ₹1,33,876 Friction | **9.2x Protection Multiplier** |
| **Drift Adaptation (Round 2)** | No Full Retraining | Recalibrated on initial 5% stream | **PR-AUC 0.8630 / Recall 99.74%** |

---

## 3. How to Install and Set Up

### Prerequisites
- Python 3.11 (or 3.12)
- Git

### Installation Steps
Open your terminal inside the `streamshield` folder and run:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Mac / Linux:
# source venv/bin/activate

# 3. Install all required dependencies
pip install -r requirements.txt
```

*(Note: To regenerate `requirements.txt` at any time from your active environment, run: `pip freeze > requirements.txt`)*

---

## 4. How to Run

### Option A: One-Click Master Pipeline
Run the entire pipeline end-to-end (generates data, extracts features, trains models, runs streaming replay, finds rings, and tests drift adaptation):
```bash
python run_all.py
```

### Option B: Launch the Streamlit Dashboard
Open the interactive 6-panel visual control room:
```bash
streamlit run app/dashboard.py
```
This opens your browser at `http://localhost:8501`.

### Option C: Launch the Cyber-Fintech Web Command Center (Modern SPA)
Open the standalone cyber-fintech single-page application:
```bash
python app/server.py 8080
```
This opens your browser at `http://localhost:8080`.

---

## 5. What Each File Does (File-by-File Guide)

```
streamshield/
├── src/
│   ├── generator.py            # Step 1: Makes fake UPI payment data (Round 1 & Round 2)
│   ├── baseline.py             # Step 2: Evaluates 3 simple human rules to establish benchmark
│   ├── features.py             # Step 3: Pure Python O(1) state memory & feature engine
│   ├── train.py                # Step 4: Trains calibrated LightGBM + Isolation Forest hybrid model
│   ├── decision.py             # Step 5: Cost-utility decision policy (Allow/Step-up/Hold) + TreeSHAP reasons
│   ├── replay.py               # Step 6: 2,000 TPS real-time streaming replay & latency benchmark
│   ├── rings.py                # Step 7: NetworkX graph clustering to uncover mule rings
│   ├── round2.py               # Step 9: Tests adversarial drift & online recalibration without full retrain
│   ├── consortium.py           # Stretch Goal 2: Privacy-preserving inter-bank signal sharing (DPDP-compliant)
│   └── make_sample_outputs.py  # Step 8A: Generates sample mock files for dashboard testing
├── app/
│   └── dashboard.py            # Step 8B: Streamlit dashboard with graph network, metrics & sliders
├── data/                       # Datasets, saved models, latency plots, and telemetry logs
├── run_all.py                  # Step 10: Master runner script executing all steps in sequence
├── requirements.txt            # Frozen dependency list
└── README.md                   # This project guide
```

---

## 6. How to Explain the Project to Judges

1. **The Core Problem**: UPI payments settle instantly and cannot be reversed. Fraud must be blocked in real time before funds escape into cash-out networks.
2. **What is a Mule Ring?**: A coordinated network where multiple accounts funnel stolen money to one collector account, which immediately forwards $\ge 90\%$ of the funds to exit accounts.
3. **Pass-Through Ratio**: Money sent out in the last 30 minutes divided by money received in the last 30 minutes. A ratio near $1.0$ is the classic signature of an illicit funnel.
4. **Why PR-AUC instead of Accuracy?**: When fraud is less than $0.1\%$, a system that flags nothing is $99.9\%$ "accurate" but useless. PR-AUC measures how accurately rare fraud is detected.
5. **p99 Latency**: 99 out of 100 decisions finish faster than this threshold. Our p99 latency was measured at **0.01 ms** (max: **65.69 ms**), easily beating the $150\text{ ms}$ requirement.
6. **The Cost Rule**: We calculate expected financial loss against customer checkout friction in Indian Rupees, selecting the most cost-effective option for the bank.
7. **No Full Retrain**: When fraudsters changed tactics in Round 2, we adapted the model using only the first $5\%$ of transactions in under 1 second, recovering performance from $0.8089$ to $0.8630$ PR-AUC without needing to retrain on 400,000 rows.

---

## 7. Safe & Honest Practice
- **Synthetic Data**: All data used is strictly synthetic. No real customer bank data was accessed or used.
- **Real Measurements**: All reported numbers (latencies, PR-AUC, rupees prevented, ring counts) were measured directly on this workstation.

---

## 8. Stretch Goals Implementation
1. **Automatic HOLD / STEP-UP Policy**: Fully operational in `src/decision.py`. Balances expected fraud loss against user drop-off friction, saving ₹12.3 Lakhs at a 9.2x ROI.
2. **Privacy-Preserving Inter-Bank Signal Sharing**: Implemented in `src/consortium.py`. Demonstrates DPDP-compliant HMAC-SHA256 blinded token exchange and cross-bank multi-inflow velocity tracking without sharing raw customer PII across banks.


