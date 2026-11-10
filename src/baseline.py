"""
StreamShield - Step 2: Simple Heuristic Rules (The Baseline)
Establishes the benchmark to beat using 3 simple rules:
(a) Brand-new account (< 7 days old) sending a large amount.
(b) Account receiving money from 5+ different senders in the last 10 minutes.
(c) First-time receiver getting a large amount.

Evaluates on the 30% time-split test period:
- PR-AUC (Average Precision)
- Recall at 1% False Alarm Rate (FPR)
"""

import os
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# ==========================================
# THRESHOLDS & PARAMETERS
# ==========================================
DATA_PATH = os.path.join("data", "round1.csv")
TRAIN_RATIO = 0.70  # First 70% time is train/warmup, last 30% is test

# Rule Parameters
LARGE_AMOUNT_THRESHOLD = 15000.0  # INR considered a large transaction
SEVEN_DAYS_SECONDS = 7 * 24 * 3600  # 604,800 seconds
TEN_MINUTES_SECONDS = 10 * 60     # 600 seconds
MULE_FAN_IN_THRESHOLD = 5         # 5 or more distinct incoming senders in 10 mins


def load_and_split_data(path, train_ratio=0.70):
    """
    Loads dataset and splits by time (chronologically).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run src/generator.py first.")
    
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Calculate time threshold for the 70/30 chronological split
    min_time = df["time"].min()
    max_time = df["time"].max()
    split_time = min_time + train_ratio * (max_time - min_time)
    
    print(f"Total rows: {len(df):,}")
    print(f"Time range: {min_time}s to {max_time}s (14 days)")
    print(f"Split time: {split_time:.0f}s (first {train_ratio*100:.0f}% train, last {(1-train_ratio)*100:.0f}% test)\n")
    
    return df, split_time


def compute_baseline_scores(df):
    """
    Simulates live stream in chronological order and assigns risk scores.
    Maintains stateful memory using deque and dict for high efficiency.
    """
    print("Evaluating 3 simple heuristic rules across all payments...")
    
    # In-memory account state
    account_first_seen = {}                 # account -> first seen timestamp
    receiver_inflow_window = defaultdict(deque)  # receiver -> deque of (timestamp, sender)
    known_receivers = set()                 # set of accounts that have received before
    
    scores = np.zeros(len(df), dtype=np.float32)
    
    times = df["time"].values
    senders = df["sender"].values
    receivers = df["receiver"].values
    amounts = df["amount"].values
    
    for i in range(len(df)):
        t = times[i]
        s = senders[i]
        r = receivers[i]
        amt = amounts[i]
        
        # Track first seen time for sender
        if s not in account_first_seen:
            account_first_seen[s] = t
        sender_age = t - account_first_seen[s]
        
        # Track first seen time for receiver
        if r not in account_first_seen:
            account_first_seen[r] = t
            
        # ----------------------------------------------------
        # Rule (a): Very new account (< 7 days old) sending a large amount
        # ----------------------------------------------------
        if sender_age < SEVEN_DAYS_SECONDS and amt >= LARGE_AMOUNT_THRESHOLD:
            # Scale score based on amount size (up to 0.85)
            score_a = min(0.85, 0.4 + (amt / 50000.0) * 0.45)
        else:
            score_a = 0.0
            
        # ----------------------------------------------------
        # Rule (b): Account received money from 5+ senders in last 10 minutes
        # ----------------------------------------------------
        inflow = receiver_inflow_window[r]
        # Evict transactions older than 10 minutes (600s)
        cutoff = t - TEN_MINUTES_SECONDS
        while inflow and inflow[0][0] < cutoff:
            inflow.popleft()
            
        # Count unique senders in this 10-minute window
        unique_senders = len(set(sender_id for _, sender_id in inflow))
        if unique_senders >= MULE_FAN_IN_THRESHOLD:
            # High suspicion of being a mule collector account
            score_b = min(1.0, 0.6 + 0.05 * (unique_senders - MULE_FAN_IN_THRESHOLD))
        else:
            score_b = 0.0
            
        # ----------------------------------------------------
        # Rule (c): First-time receiver with a large amount
        # ----------------------------------------------------
        if (r not in known_receivers) and (amt >= LARGE_AMOUNT_THRESHOLD):
            # Suspicion of social engineering scam or new burner drop account
            score_c = min(0.90, 0.5 + (amt / 50000.0) * 0.4)
        else:
            score_c = 0.0
            
        # The final risk score is the highest of the three rules
        scores[i] = max(score_a, score_b, score_c)
        
        # Update state with current transaction
        inflow.append((t, s))
        known_receivers.add(r)
        
    df["risk_score"] = scores
    return df


def evaluate_test_metrics(df, split_time):
    """
    Computes PR-AUC and Recall at 1% False Positive Rate on the test period only.
    """
    test_mask = df["time"] >= split_time
    df_test = df[test_mask].copy()
    
    y_true = df_test["is_fraud"].values
    y_scores = df_test["risk_score"].values
    
    total_test = len(y_true)
    fraud_test = int(np.sum(y_true))
    fraud_rate_pct = (fraud_test / total_test) * 100.0
    
    # 1. PR-AUC (Average Precision)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # 2. Recall at 1% False Positive Rate (FPR)
    # 1% false alarms means only 1% of legitimate payments are wrongly flagged
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    
    # Find the threshold at the 99th percentile of negative scores
    threshold_1pct_fpr = np.percentile(neg_scores, 99.0)
    
    # If the threshold is 0.0 (e.g. >99% have score 0), find smallest non-zero threshold
    if threshold_1pct_fpr == 0.0:
        non_zero = neg_scores[neg_scores > 0]
        if len(non_zero) > 0:
            threshold_1pct_fpr = np.min(non_zero)
        else:
            threshold_1pct_fpr = 0.01
            
    # Recall = percentage of true frauds flagged above this threshold
    caught_fraud = np.sum(pos_scores >= threshold_1pct_fpr)
    recall_at_1pct_fpr = (caught_fraud / len(pos_scores)) * 100.0 if len(pos_scores) > 0 else 0.0
    
    return {
        "total_test": total_test,
        "fraud_test": fraud_test,
        "fraud_rate_pct": fraud_rate_pct,
        "pr_auc": pr_auc,
        "recall_at_1pct_fpr": recall_at_1pct_fpr,
        "threshold": threshold_1pct_fpr
    }


def print_results(metrics):
    """
    Prints the benchmark results table and plain-language explanation.
    """
    print("=" * 65)
    print(" STEP 2: SIMPLE RULES BASELINE RESULTS (TEST PERIOD)")
    print("=" * 65)
    print(f"Test Transactions Evaluated : {metrics['total_test']:,}")
    print(f"Actual Fraud in Test Period : {metrics['fraud_test']:,} ({metrics['fraud_rate_pct']:.3f}%)")
    print("-" * 65)
    print(f"{'Metric':<35} | {'Baseline (Simple Rules)':<25}")
    print("-" * 65)
    print(f"{'PR-AUC (Average Precision)':<35} | {metrics['pr_auc']:.4f}")
    print(f"{'Recall @ 1% False Alarms':<35} | {metrics['recall_at_1pct_fpr']:.2f}%")
    print("=" * 65)
    
    print("\n--- What these two numbers mean in simple words ---")
    print("1. PR-AUC (Average Precision):")
    print("   In fraud detection, standard accuracy is misleading because 99.9% of payments")
    print("   are good (a 'dumb' system that approves everything is 99.9% accurate but catches 0 fraud).")
    print(f"   PR-AUC scores how well we find rare fraud from 0 to 1. A score of {metrics['pr_auc']:.4f}")
    print("   shows that simple manual rules catch some obvious fraud, but miss subtle rings.")
    print("\n2. Recall @ 1% False Alarms:")
    print("   If a bank allows only 1 out of every 100 good customers to be annoyed by an extra check,")
    print(f"   these 3 simple rules catch {metrics['recall_at_1pct_fpr']:.1f}% of the stolen money.")
    print("   Our upcoming smart machine learning model in Step 4 must beat these two numbers!\n")


def main():
    df, split_time = load_and_split_data(DATA_PATH, TRAIN_RATIO)
    df = compute_baseline_scores(df)
    metrics = evaluate_test_metrics(df, split_time)
    print_results(metrics)


if __name__ == "__main__":
    main()
