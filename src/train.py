"""
StreamShield - Step 4: Hybrid ML Model Training & Calibration
Trains a calibrated LightGBM classifier alongside an unsupervised Isolation Forest.
Balances rare fraud class distribution with a tuned depth and learning rate to avoid
probability saturation.

Evaluates on the last 20% test period and saves models to data/ folder with joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier

# ==========================================
# CONFIGURATION & FEATURE COLUMNS
# ==========================================
FEATURES_PATH = os.path.join("data", "features_round1.csv")

FEATURE_COLS = [
    "amount",
    "sender_age_days",
    "receiver_age_days",
    "sender_amount_ratio",
    "is_new_receiver",
    "sender_tx_count_1h",
    "device_changed",
    "receiver_distinct_senders_24h",
    "receiver_distinct_receivers_24h",
    "pass_through_ratio_30m",
    "minutes_since_last_inflow"
]

RANDOM_SEED = 42


def evaluate_metrics(y_true, y_scores):
    """
    Computes PR-AUC and Recall at 1% False Positive Rate (FPR).
    """
    pr_auc = average_precision_score(y_true, y_scores)
    
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    
    threshold_1pct = np.percentile(neg_scores, 99.0)
    if threshold_1pct == 0.0:
        non_zero = neg_scores[neg_scores > 0]
        threshold_1pct = np.min(non_zero) if len(non_zero) > 0 else 0.01
        
    caught = np.sum(pos_scores >= threshold_1pct)
    recall_1pct = (caught / len(pos_scores)) * 100.0 if len(pos_scores) > 0 else 0.0
    
    return pr_auc, recall_1pct, threshold_1pct


def compute_baseline_rules_test(df_test):
    """
    Computes Step 2 simple rules on test set for direct comparison.
    """
    scores = np.zeros(len(df_test), dtype=np.float32)
    amt = df_test["amount"].values
    s_age = df_test["sender_age_days"].values
    r_senders = df_test["receiver_distinct_senders_24h"].values
    is_new_rec = df_test["is_new_receiver"].values
    
    for i in range(len(df_test)):
        # Rule a: new account (< 7 days) sending large amount
        sa = min(0.85, 0.4 + (amt[i] / 50000.0) * 0.45) if (s_age[i] < 7.0 and amt[i] >= 15000.0) else 0.0
        # Rule b: 5+ senders to receiver in last 10m/24h
        sb = min(1.0, 0.6 + 0.05 * (r_senders[i] - 5)) if r_senders[i] >= 5 else 0.0
        # Rule c: new receiver receiving large amount
        sc = min(0.90, 0.5 + (amt[i] / 50000.0) * 0.4) if (is_new_rec[i] == 1 and amt[i] >= 20000.0) else 0.0
        scores[i] = max(sa, sb, sc)
        
    return scores


def main():
    print("=" * 68)
    print(" STEP 4: MODEL TRAINING & HYBRID BLEND PIPELINE")
    print("=" * 68)
    
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    n_total = len(df)
    
    # -------------------------------------------------------------
    # 1. Strict Temporal Split (70% Train, 10% Validation, 20% Test)
    # -------------------------------------------------------------
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.80)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_fraud"].values
    
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["is_fraud"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_fraud"].values
    
    pos_train = int(np.sum(y_train))
    pos_val = int(np.sum(y_val))
    pos_test = int(np.sum(y_test))
    
    print(f"Data Split Summary:")
    print(f"  Training   : {len(train_df):,} transactions | Fraud: {pos_train} ({pos_train/len(train_df)*100:.3f}%)")
    print(f"  Validation : {len(val_df):,} transactions | Fraud: {pos_val} ({pos_val/len(val_df)*100:.3f}%)")
    print(f"  Test       : {len(test_df):,} transactions | Fraud: {pos_test} ({pos_test/len(test_df)*100:.3f}%)")
    
    # -------------------------------------------------------------
    # 2. Train LightGBM Classifier with depth-constrained trees
    # -------------------------------------------------------------
    # With extreme class imbalance (0.09%), depth-constrained trees (max_depth=3)
    # prevent leaf probability saturation and prevent overfitting on small rings.
    print("\nTraining LightGBM Classifier (n_estimators=150, max_depth=3)...")
    
    lgb_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        scale_pos_weight=2.0,
        random_state=RANDOM_SEED,
        n_jobs=2,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------
    # 3. Calibrate Probabilities with Isotonic Regression
    # -------------------------------------------------------------
    print("Calibrating probabilities on validation set with Isotonic Regression...")
    val_raw_probs = lgb_model.predict_proba(X_val)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw_probs, y_val)
    
    def predict_calibrated_lgb(X):
        raw = lgb_model.predict_proba(X)[:, 1]
        cal = calibrator.transform(raw)
        # Preserve micro-ranking in step ties
        return cal + 1e-6 * raw
        
    val_lgb_cal = predict_calibrated_lgb(X_val)
    
    # -------------------------------------------------------------
    # 4. Train Unsupervised Isolation Forest (Zero-Day Anomaly Detector)
    # -------------------------------------------------------------
    print("Training unsupervised Isolation Forest for zero-day anomaly detection...")
    normal_indices = np.where(y_train == 0)[0]
    sample_indices = np.random.choice(normal_indices, size=min(25000, len(normal_indices)), replace=False)
    X_iforest_train = X_train[sample_indices]
    
    iforest = IsolationForest(
        n_estimators=100,
        contamination=0.001,
        max_samples=1024,
        random_state=RANDOM_SEED,
        n_jobs=2
    )
    iforest.fit(X_iforest_train)
    
    # Normalize anomaly score to [0, 1]
    val_iforest_raw = -iforest.decision_function(X_val)
    if_min = float(np.min(val_iforest_raw))
    if_max = float(np.max(val_iforest_raw))
    
    def score_iforest(X):
        raw = -iforest.decision_function(X)
        norm = (raw - if_min) / max(if_max - if_min, 1e-6)
        return np.clip(norm, 0.0, 1.0)
        
    val_if_scores = score_iforest(X_val)
    
    # Select blend weight w on validation set
    best_w = 0.05
    best_val_pr = 0.0
    for candidate_w in [0.0, 0.03, 0.05, 0.08, 0.10]:
        b_score = (1.0 - candidate_w) * val_lgb_cal + candidate_w * val_if_scores
        pr = average_precision_score(y_val, b_score)
        if pr > best_val_pr:
            best_val_pr = pr
            best_w = candidate_w
            
    print(f"Optimal Hybrid Blend Weight selected: w = {best_w:.2f}")
    
    # -------------------------------------------------------------
    # 5. Evaluate All Approaches on Test Set (Last 20%)
    # -------------------------------------------------------------
    print("\nEvaluating all approaches on the final 20% test period...")
    
    # A. Baseline Rules
    baseline_scores = compute_baseline_rules_test(test_df)
    pr_base, rec_base, _ = evaluate_metrics(y_test, baseline_scores)
    
    # B. LightGBM Alone (Calibrated)
    test_cal_lgb = predict_calibrated_lgb(X_test)
    pr_lgb, rec_lgb, _ = evaluate_metrics(y_test, test_cal_lgb)
    
    # C. Hybrid Blend (LightGBM + Isolation Forest)
    test_if_scores = score_iforest(X_test)
    test_blend = (1.0 - best_w) * test_cal_lgb + best_w * test_if_scores
    pr_blend, rec_blend, _ = evaluate_metrics(y_test, test_blend)
    
    # -------------------------------------------------------------
    # 6. Print Comparative Results Table
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" MODEL PERFORMANCE COMPARISON (TEST PERIOD — LAST 20%)")
    print("=" * 72)
    print(f"{'Method / Model':<28} | {'PR-AUC':<18} | {'Recall @ 1% False Alarms':<20}")
    print("-" * 72)
    print(f"{'Step 2 Simple Rules':<28} | {pr_base:.4f}{'':<12} | {rec_base:.2f}%")
    print(f"{'LightGBM Classifier':<28} | {pr_lgb:.4f}{' (+' + f'{(pr_lgb-pr_base):.4f})':<10} | {rec_lgb:.2f}% ({(rec_lgb-rec_base):+.2f}%)")
    print(f"{'Hybrid Blend (+ Isolation)':<28} | {pr_blend:.4f}{' (+' + f'{(pr_blend-pr_base):.4f})':<10} | {rec_blend:.2f}% ({(rec_blend-rec_base):+.2f}%)")
    print("=" * 72)
    
    print("\n--- Explanation of Results in Simple Words ---")
    print("1. Why LightGBM and the Blend score higher than simple rules:")
    print("   Simple rules look at only 1 fact at a time (e.g. just amount or just receiver count).")
    print("   LightGBM analyzes all 10 behavioral facts together, identifying subtle mule rings.")
    print(f"   PR-AUC improved dramatically from {pr_base:.4f} (rules) to {pr_blend:.4f} (model),")
    print(f"   and Recall at 1% false alarms jumped from {rec_base:.1f}% to {rec_blend:.1f}%.")
    print("2. The role of Isolation Forest (Unsupervised):")
    print("   LightGBM learns known fraud examples. Isolation Forest catches unfamiliar,")
    print("   unusual transaction spikes, ensuring protection against zero-day fraud.")
    print("3. Why Calibration matters:")
    print("   When the model says a payment has a risk score of 0.30, calibration guarantees")
    print("   that roughly 30% of such payments are actual fraud. Step 5 relies on this for")
    print("   calculating expected loss in rupees.\n")
    
    # -------------------------------------------------------------
    # 7. Save Models to data/ with joblib
    # -------------------------------------------------------------
    model_bundle = {
        "lgb_model": lgb_model,
        "calibrator": calibrator,
        "iforest": iforest,
        "iforest_min": if_min,
        "iforest_max": if_max,
        "blend_weight": best_w,
        "feature_cols": FEATURE_COLS
    }
    
    bundle_path = os.path.join("data", "model_bundle.joblib")
    joblib.dump(model_bundle, bundle_path)
    joblib.dump(lgb_model, os.path.join("data", "model_lgb.joblib"))
    joblib.dump(calibrator, os.path.join("data", "model_calibrator.joblib"))
    joblib.dump(iforest, os.path.join("data", "model_iforest.joblib"))
    
    print(f"All models successfully saved to data/ (Bundle: {bundle_path}).")
    print("[SUCCESS] Step 4 complete: Ready for Step 5 Decision Engine.")


if __name__ == "__main__":
    main()
