# StreamShield Architecture & Flowcharts
### Real-Time Mule-Ring & Scam Detection for UPI Payments
**Innovasia 2026 · PS-01 · Fintech Hackathon**

---

## 1. Overall Architecture
The system operates in two distinct lanes: **Lane A (Offline Preparation & Training)** executed prior to deployment, and **Lane B (Online Live Scoring & Interception)** processing transactions in real time.

```mermaid
flowchart TD
    subgraph Offline["Lane A: Offline Pipeline (Trained Once)"]
        G["Synthetic Data Generator<br/>(src/generator.py)"] --> R1["round1.csv<br/>round2.csv"]
        R1 --> FB["Feature Builder<br/>(src/features.py)"]
        FB --> F1["features_round1.csv<br/>features_round2.csv"]
        F1 --> TR["Train + Calibrate<br/>LightGBM + Isolation Forest<br/>(src/train.py)"]
        TR --> MD["Saved Models Bundle<br/>(data/model_bundle.joblib)"]
        R1 -.-> BL["Baseline Rules Benchmark<br/>(src/baseline.py)"]
    end

    subgraph Online["Lane B: Online Real-Time Scoring Stream"]
        RP["Replay Streamer 2,000 TPS<br/>(src/replay.py)"] --> RT["Account Router<br/>(Hash Modulo)"]
        RT --> SW["Scoring Worker<br/>• Account Memory (RAM)<br/>• Feature Extraction<br/>• Model Scoring<br/>• Cost Decision Policy"]
        MD -.->|Loaded once| SW
        SW --> DEC["data/decisions.jsonl<br/>(Every Scored Payment)"]
        SW --> MET["data/metrics.json<br/>(Latency & PR-AUC)"]
        DEC --> RD["Ring Detector<br/>(src/rings.py)"]
        RD --> RNG["data/rings.json<br/>(Clustered Rings)"]
    end

    subgraph UI["Visual Control Room"]
        DEC --> DASH["Live Operations Dashboard<br/>(Streamlit / Cyber SPA)"]
        RNG --> DASH
        MET --> DASH
    end
```

---

## 2. Offline Pipeline: From Data to Trained Model
The offline training loop strictly enforces chronological time splits to prevent future data leakage. The model is saved only if it significantly outperforms simple baseline rules.

```mermaid
flowchart TD
    S([START]) --> G["Generate Synthetic Data<br/>(Round 1 & Round 2)"]
    G --> SP["Split Chronologically by TIME<br/>Train: 70% | Valid: 10% | Test: 20%"]
    SP --> BF["Build Features<br/>(Using strictly past events)"]
    BF --> CSV["features_round1.csv"]
    CSV --> TL["Train LightGBM<br/>(rare-fraud class weight)"]
    CSV --> TI["Train Isolation Forest<br/>(unsupervised anomaly detection)"]
    TL --> CAL["Isotonic Probability Calibration<br/>(on 10% validation split)"]
    CAL & TI --> BLD["Blend Scores<br/>(1-w) * LightGBM + w * IsolationForest"]
    BLD --> EVAL["Evaluate on 20% Test Period<br/>PR-AUC & Recall @ 1% False Alarms"]
    
    SP -.-> BR["Evaluate Baseline Rules<br/>(3 heuristic rules)"]
    BR --> CHK{"Better than<br/>Baseline?"}
    EVAL --> CHK
    
    CHK -->|YES| SAVE["Save Models with Joblib<br/>(data/model_bundle.joblib)"]
    CHK -->|NO| DBG["Check for Data Leakage<br/>or Feature Code Mismatch"]
    
    SAVE --> R2["Round 2 Drift Test<br/>(Score, Recalibrate on 5%)"]
    R2 --> RES([Slide-Ready Measured Numbers])
```

---

## 3. One Payment, One Decision (Sub-150ms Live Path)
Every transaction follows this strict sequence from network arrival to automated three-way resolution (`ALLOW`, `STEP-UP`, `HOLD`).

```mermaid
flowchart TD
    TX([Payment Arrives<br/>sender, receiver, amount, device, time]) --> RT["Router<br/>hash(sender) -> worker"]
    RT --> RM["Read In-Memory State<br/>(sender & receiver)"]
    RM --> BF["Build 10 Features<br/>(behaviour + velocity + pass-through)"]
    BF --> SC["Score Ensemble<br/>LightGBM + Isolation Forest"]
    SC --> CL["Blend + Calibrate<br/>Empirical Fraud Probability p"]
    
    CL --> CC["Cost-Utility Check<br/>Loss = p * amount * (1 - recovery)<br/>Friction = step_up_cost + (1 - p) * abandon * amount"]
    
    CC --> C1{"Loss >= Hold Threshold<br/>(Rs 2,000)?"}
    C1 -->|YES| HOLD["🔴 HOLD<br/>(Block Payment Immediately)"]
    C1 -->|NO| C2{"Loss >= Friction?"}
    
    C2 -->|YES| STEP["🟡 STEP-UP<br/>(Biometric / OTP Challenge)"]
    C2 -->|NO| ALLOW["🟢 ALLOW<br/>(Frictionless Pass-Through)"]
    
    ALLOW --> WD["Write Decision Record<br/>to decisions.jsonl"]
    STEP & HOLD --> SHAP["Top 3 TreeSHAP Reasons<br/>(pred_contrib to plain English)"]
    SHAP --> WD
    
    WD --> UM["Update In-Memory State<br/>(record this payment in RAM)"]
    UM --> NXT([Next Payment in Stream])
```

---

## 4. Account Memory & Feature Extraction
StreamShield maintains an $O(1)$ in-memory state per account using pure Python data structures. No graph database roundtrips occur in the live decision path.

| In-Memory Structure (per Account) | Storage Type | Rolling Window | Features Calculated From It | Category |
| :--- | :--- | :--- | :--- | :--- |
| `first_seen` | Timestamp | Permanent | **Account Age (days)** | Behavioral |
| `amount_history` | Running sum & count | Permanent | **Amount Compared to Usual Ratio** | Behavioral |
| `known_payees` | Python `set` | Permanent | **Is Receiver New for Sender? (0/1)** | Network |
| `last_device` | Device ID | Latest | **Device Changed? (0/1)** | Security |
| `out_events_1h` | Python `deque` | 1 Hour | **Payments Made in Last 1 Hour** | Velocity |
| `in_events_24h` | Python `deque` | 24 Hours | **Distinct Senders Received From (24h)** | Network / Fan-In |
| `out_events_24h` | Python `deque` | 24 Hours | **Distinct Receivers Paid (24h)** | Network / Fan-Out |
| `inflow_30m` & `outflow_30m` | Python `deque` | 30 Minutes | **Pass-Through Ratio (30m)** | **Mule Indicator** |
| `last_inflow_time` | Timestamp | Latest | **Minutes Since Last Incoming Payment** | Velocity |

---

## 5. Mule Ring Detection Flow
Single transactions may appear harmless. Coordinated mule rings are uncovered when high-risk payments are connected across the counterparty graph.

```mermaid
flowchart LR
    DEC["decisions.jsonl<br/>(All Scored Payments)"] --> FILT["Filter Suspicious Payments<br/>(HOLD, STEP-UP, or Pass-Through > 0.8)"]
    FILT --> ACC["Extract Transacting Accounts<br/>(Senders & Receivers)"]
    ACC --> GRP["Join Accounts via Connected Components<br/>(NetworkX Graph Clustering)"]
    GRP --> CHK{"Group has >= 3<br/>Accounts?"}
    CHK -->|YES| RING["Create Mule Ring Dossier<br/>• ring_id<br/>• member accounts<br/>• total rupees funnelled<br/>• time first flagged"]
    CHK -->|NO| IGN["Ignore Small Benign Pair"]
    RING --> OUT["Save to data/rings.json<br/>& Render Red in Dashboard"]
    RING --> VAL["Benchmark Against True Ring IDs<br/>• Ring Recall: 5 of 5 (100%)<br/>• Detection Delay: 98.0 seconds"]
```

---

## 6. Round 2: Adversarial Concept Drift Adaptation
When fraudsters alter tactics (delaying payouts by 30–120m, smurfing into smaller chunks, daisy chains), StreamShield adapts without full retraining.

```mermaid
flowchart TD
    R2D["Round 2 Data Stream<br/>(Delayed payouts, smurfing, chains)"] --> SC1["Score with Round 1 Models<br/>(No retraining yet)"]
    SC1 --> M1["Measure Degradation<br/>PR-AUC drops to 0.2418"]
    M1 --> SAMPLE["Take First 5% of Stream<br/>(Small adaptation window: 13 fraud cases)"]
    SAMPLE --> ADAPT["Online Recalibration:<br/>1. Adjust blend weight w<br/>2. Re-fit Isotonic Calibration curve<br/>3. Evaluated on remaining 95% in 1.17s"]
    ADAPT --> SC2["Score Remaining 95% Stream<br/>with Updated Settings"]
    SC2 --> REC["Decision-Level Defense Stabilized!<br/>False Alarms: 0.06% | Net Benefit Positive"]
```

---

## 7. Speed, Scaling & Latency Budget
Meeting the $150\text{ ms}$ p99 SLA at $\ge 2,000\text{ TPS}$ on a single machine:

```mermaid
flowchart LR
    A["Batch Wait<br/>(10 ms)"] --> B["Feature Extraction<br/>(20 ms)"]
    B --> C["Model Scoring<br/>(40 ms)"]
    C --> D["Cost Decision<br/>(10 ms)"]
    D --> E["Safety Headroom Buffer<br/>(70 ms)"]
```

**Total Latency Budget:** $10 + 20 + 40 + 10 + 70 = 150\text{ ms}$ (Target: $\le 150\text{ ms}$).  
**Measured Performance:** Average batch turnaround = **$31.5\text{ ms}$**; p99 turnaround = **$109.8\text{ ms}$**.

---

## 8. Dashboard Data Contract
The dashboard operates decoupled from the scoring engine, reading strictly three standardized data contracts:

```mermaid
flowchart TD
    subgraph Files["Data Contract Files (data/)"]
        DJ["decisions.jsonl<br/>• time, txn_id<br/>• sender, receiver, amount<br/>• decision, score, reasons<br/>• ring_id, latency_ms"]
        RJ["rings.json<br/>• ring_id<br/>• member accounts<br/>• total rupees funnelled<br/>• first_flagged_time"]
        MJ["metrics.json<br/>• p50, p95, p99 latency<br/>• PR-AUC (rules vs model)<br/>• rupees prevented & friction"]
    end

    subgraph Panels["Dashboard Visual Panels"]
        DJ --> P1["1. Live Decision Feed (Color Badges + Reasons)"]
        RJ --> P2["2. Interactive 2D Mule Ring Graph"]
        MJ --> P3["3. Financial KPI Counters (Rs Saved vs Friction)"]
        MJ --> P4["4. Latency SLA Histogram (150ms line)"]
        MJ --> P5["5. Baseline Rules vs Model Benchmark Table"]
        Panels --> P6["6. Interactive Cost Sliders (Live Recalculation)"]
    end
```

---

## 9. Real-World Banking Deployment (Future Vision)
In production, StreamShield deploys directly inside the issuing/acquiring bank's secure perimeter alongside the UPI payment switch.

```mermaid
flowchart TD
    subgraph Perimeter["Inside Bank's Secure Perimeter (Zero PII Leaves)"]
        SW["UPI Payment Switch<br/>(Pre-Settlement Authorization)"]
        
        subgraph SS["StreamShield Core Engine"]
            MEM["In-Memory State<br/>(Features)"]
            ML["Hybrid Ensemble<br/>(LightGBM + Anomaly)"]
            DEC["Cost-Utility Policy<br/>(ALLOW / STEP-UP / HOLD)"]
            EXP["TreeSHAP Reasons +<br/>Ring Cluster Engine"]
        end
        
        SW -->|Payment Event| SS
        DEC -->|Authorization Decision| SW
        
        SS --> DISP["Dispute Officer Console<br/>(Audit Trails & Ring Alerts)"]
        SS --> USER["Customer Notification<br/>(Step-Up Challenge / Hold Notice)"]
    end

    subgraph Consortium["Inter-Bank Consortium (Stretch Goal 2)"]
        SS -.->|Salted HMAC Tokens Only| CB["Privacy-Preserving Consortium Ledger<br/>(DPDP-Compliant Cross-Bank Velocity)"]
        CB -.->|Multi-Bank Alert| SS
    end
```

---

## 10. File Map: Reads, Writes & Roles

| # | File | Reads From | Writes To | Core Role in Pipeline |
|:---:|:---|:---|:---|:---|
| **1** | [`src/generator.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/generator.py) | Configuration constants | `data/round1.csv`, `data/round2.csv` | Generates 800k+ realistic UPI payments with planted rings and scams |
| **2** | [`src/baseline.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/baseline.py) | `data/round1.csv` | Terminal benchmark table | Evaluates 3 simple human rules to establish baseline PR-AUC (`0.1933`) |
| **3** | [`src/features.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/features.py) | `round1.csv`, `round2.csv` | `features_round1.csv`, `features_round2.csv` | $O(1)$ in-memory feature engine (pass-through ratio, 24h network velocity) |
| **4** | [`src/train.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/train.py) | `features_round1.csv` | `data/model_bundle.joblib` | Trains calibrated LightGBM + Isolation Forest hybrid model (`PR-AUC = 0.6076`) |
| **5** | [`src/decision.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/decision.py) | Saved model bundle | Decision evaluation functions | Economic cost policy (expected loss vs friction) + TreeSHAP plain reasons |
| **6** | [`src/replay.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/replay.py) | Test transactions, models | `decisions.jsonl`, `metrics.json`, `latency.png` | 2,000 TPS streaming replay measuring p50, p95, p99 decision latency |
| **7** | [`src/rings.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/rings.py) | `data/decisions.jsonl` | `data/rings.json` | NetworkX connected-components clustering detecting 100% of mule rings |
| **8** | [`app/dashboard.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/app/dashboard.py) | `decisions.jsonl`, `rings.json`, `metrics.json` | Streamlit interactive UI | 6-panel live operations dashboard for evaluators |
| **9** | [`src/round2.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/src/round2.py) | `features_round2.csv`, models | 3-row evaluation table | Demonstrates sub-second online recalibration on 5% sample (no full retrain) |
| **10** | [`run_all.py`](file:///c:/Users/syedm/OneDrive/Desktop/streamshield/run_all.py) | All component modules | Full project verification | One-click master orchestrator running all 6 stages sequentially |

