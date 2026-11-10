"""
StreamShield - Step 5: Cost-Utility Decision Engine & Explainability
Evaluates expected financial loss vs customer friction to assign:
- ALLOW
- STEP-UP
- HOLD

Generates up to 3 plain-English reasons for STEP-UP and HOLD decisions using
LightGBM TreeSHAP feature contributions.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ==========================================
# BUSINESS SETTINGS (Change to tune policy)
# ==========================================
RECOVERY_RATE = 0.5       # Share of stolen money we expect to get back
STEP_UP_COST = 5.0        # Rupee cost of customer verification (OTP/biometrics)
ABANDON_PROB = 0.05       # 5% chance a good customer gives up if challenged
HOLD_THRESHOLD = 2000.0   # Expected loss in rupees above which we immediately HOLD

FEATURES_PATH = os.path.join("data", "features_round1.csv")
MODEL_BUNDLE_PATH = os.path.join("data", "model_bundle.joblib")


def format_feature_reason(col_name, val):
    """
    Turns a feature name and value into a clear, plain-English sentence.
    """
    if col_name == "receiver_distinct_senders_24h":
        return f"Received money from {int(val)} different senders in the last 24 hours"
    elif col_name == "pass_through_ratio_30m":
        pct = min(100, int(val * 100))
        return f"Sent out {pct} percent of incoming money within 30 minutes"
    elif col_name == "amount":
        return f"High payment amount of Rs {val:,.2f}"
    elif col_name == "sender_amount_ratio":
        return f"Amount is {val:.1f}x higher than this sender's normal payment"
    elif col_name == "is_new_receiver":
        return "First-time payment to a receiver this sender never paid before"
    elif col_name == "device_changed":
        return "Transaction initiated from a newly registered or unverified device"
    elif col_name == "sender_age_days":
        days = max(1, int(val))
        return f"Sender account is only {days} days old"
    elif col_name == "receiver_age_days":
        days = max(1, int(val))
        return f"Receiver account is only {days} days old"
    elif col_name == "sender_tx_count_1h":
        return f"High velocity: sender made {int(val)} payments in the last hour"
    elif col_name == "minutes_since_last_inflow":
        mins = max(1, int(val))
        return f"Rapid turnaround: received incoming funds only {mins} minutes ago"
    elif col_name == "receiver_distinct_receivers_24h":
        return f"Receiver distributed funds to {int(val)} different accounts in 24 hours"
    else:
        return f"Unusual pattern detected in {col_name}"


def evaluate_decision_and_reasons(amount, fraud_prob, feature_vector, lgb_model, feature_names):
    """
    REUSABLE FUNCTION:
    Computes expected loss, friction, decision (ALLOW, STEP-UP, HOLD), and reasons.
    Can be imported and called on single transactions during live streaming replay.
    """
    expected_loss = fraud_prob * amount * (1.0 - RECOVERY_RATE)
    friction = STEP_UP_COST + (1.0 - fraud_prob) * ABANDON_PROB * amount
    
    if expected_loss >= HOLD_THRESHOLD:
        decision = "HOLD"
    elif expected_loss >= friction:
        decision = "STEP-UP"
    else:
        decision = "ALLOW"
        
    reasons = []
    if decision in ["STEP-UP", "HOLD"]:
        # Compute TreeSHAP contributions for this transaction
        vec_2d = feature_vector.reshape(1, -1)
        contribs = lgb_model.booster_.predict(vec_2d, pred_contrib=True)[0, :-1]
        
        # Sort features by highest positive contribution (features pushing risk UP)
        sorted_indices = np.argsort(-contribs)
        
        for idx in sorted_indices:
            if len(reasons) >= 3:
                break
            feat_name = feature_names[idx]
            val = feature_vector[idx]
            # Ignore features that are 0 or negative contribution
            if contribs[idx] > 0.0 or len(reasons) == 0:
                reasons.append(format_feature_reason(feat_name, val))
                
        # Fallback if no specific feature had positive contribution
        if not reasons:
            reasons.append(f"Elevated risk score ({fraud_prob*100:.1f}%) exceeding safety policy")
            
    return decision, expected_loss, friction, reasons


def main():
    print("=" * 70)
    print(" STEP 5: COST-UTILITY DECISION ENGINE & EXPLAINABILITY")
    print("=" * 70)
    print(f"Policy Settings:")
    print(f"  Recovery Rate   : {RECOVERY_RATE*100:.0f}% of stolen money recovered")
    print(f"  Step-up Cost    : Rs {STEP_UP_COST:.2f} per challenged payment")
    print(f"  Abandon Rate    : {ABANDON_PROB*100:.0f}% customer drop-off on friction")
    print(f"  Hold Threshold  : Rs {HOLD_THRESHOLD:,.2f} expected loss\n")
    
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_BUNDLE_PATH}. Run src/train.py first.")
        
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    lgb_model = bundle["lgb_model"]
    calibrator = bundle["calibrator"]
    iforest = bundle["iforest"]
    if_min = bundle["iforest_min"]
    if_max = bundle["iforest_max"]
    w = bundle["blend_weight"]
    feature_cols = bundle["feature_cols"]
    
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Evaluate on the last 20% test period
    test_start = int(len(df) * 0.80)
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)
    n_test = len(test_df)
    
    X_test = test_df[feature_cols].values
    y_test = test_df["is_fraud"].values
    amounts = test_df["amount"].values
    
    print(f"Evaluating decisions for {n_test:,} test payments...")
    
    # 1. Compute hybrid risk scores
    raw_lgb = lgb_model.predict_proba(X_test)[:, 1]
    cal_lgb = calibrator.transform(raw_lgb)
    
    if_raw = -iforest.decision_function(X_test)
    if_scores = np.clip((if_raw - if_min) / max(if_max - if_min, 1e-6), 0.0, 1.0)
    
    hybrid_scores = (1.0 - w) * (cal_lgb + 1e-6 * raw_lgb) + w * if_scores
    hybrid_scores = np.clip(hybrid_scores, 0.0, 1.0)
    
    # 2. Vectorized batch decisioning for speed
    expected_losses = hybrid_scores * amounts * (1.0 - RECOVERY_RATE)
    frictions = STEP_UP_COST + (1.0 - hybrid_scores) * ABANDON_PROB * amounts
    
    decisions = []
    rupees_prevented = 0.0
    rupees_friction = 0.0
    
    flagged_indices = []
    
    for i in range(n_test):
        el = expected_losses[i]
        fr = frictions[i]
        amt = amounts[i]
        is_fraud = y_test[i]
        
        if el >= HOLD_THRESHOLD:
            dec = "HOLD"
        elif el >= fr:
            dec = "STEP-UP"
        else:
            dec = "ALLOW"
            
        decisions.append(dec)
        
        # Financial impact tracking
        if dec in ["STEP-UP", "HOLD"]:
            flagged_indices.append(i)
            if is_fraud == 1:
                # Fraud prevented (adjusted by what we save)
                rupees_prevented += amt * (1.0 - RECOVERY_RATE)
            else:
                # Friction caused on good customers
                if dec == "STEP-UP":
                    rupees_friction += STEP_UP_COST + ABANDON_PROB * amt
                else:
                    rupees_friction += amt
                    
    test_df["score"] = hybrid_scores
    test_df["decision"] = decisions
    
    allow_count = decisions.count("ALLOW")
    step_up_count = decisions.count("STEP-UP")
    hold_count = decisions.count("HOLD")
    
    print("=" * 70)
    print(" BUSINESS DECISION SUMMARY (TEST PERIOD)")
    print("=" * 70)
    print(f"Total Evaluated : {n_test:,}")
    print(f"ALLOW Decisions : {allow_count:,} ({allow_count/n_test*100:.2f}%)")
    print(f"STEP-UP (Verify): {step_up_count:,} ({step_up_count/n_test*100:.2f}%)")
    print(f"HOLD (Blocked)  : {hold_count:,} ({hold_count/n_test*100:.2f}%)")
    print("-" * 70)
    print(f"Rupees of Fraud Prevented : Rs {rupees_prevented:,.2f}")
    print(f"Rupees of Friction Caused : Rs {rupees_friction:,.2f}")
    roi = rupees_prevented / max(rupees_friction, 1.0)
    print(f"Net Protection Multiplier : {roi:.1f}x (saved Rs {roi:.1f} for every Rs 1 of friction)")
    print("=" * 70 + "\n")
    
    # 3. Print 10 Flagged Payments with their Plain-English Reasons
    print("=" * 70)
    print(" 10 FLAGGED PAYMENTS WITH REAL-TIME EXPLAINABILITY REASONS")
    print("=" * 70)
    
    sample_show = flagged_indices[:10]
    for idx_num, i in enumerate(sample_show):
        row = test_df.iloc[i]
        _, _, _, reasons = evaluate_decision_and_reasons(
            amounts[i], hybrid_scores[i], X_test[i], lgb_model, feature_cols
        )
        fraud_label = "TRUE FRAUD" if row['is_fraud'] == 1 else "GOOD CUSTOMER"
        print(f"[{idx_num + 1}] Txn: {row['txn_id']} | Decision: {row['decision']} | Score: {row['score']:.3f} | Rs {row['amount']:,.2f} | ({fraud_label})")
        print(f"    Sender: {row['sender']} -> Receiver: {row['receiver']}")
        print("    Reasons:")
        for r in reasons:
            print(f"      • {r}")
        print()
    print("=" * 70)
    print("[SUCCESS] Step 5 complete: Decision engine & explainability ready.")


if __name__ == "__main__":
    main()

