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


def generate_customer_message(decision):
    """
    Returns a short, reassuring customer-facing sentence.
    NEVER reveals internal model facts, risk scores, or decision thresholds.
    """
    if decision == "STEP-UP":
        return "We paused this payment to check the payee is safe. Please confirm to continue."
    elif decision == "HOLD":
        return "This transfer was temporarily placed on hold for your account security. Our support team is reviewing it."
    else:
        return "Payment processed successfully."


# Typical values seen in legitimate (ALLOW) payments
TYPICAL_ALLOW_VALUES = {
    "sender_age_days": 180.0,
    "receiver_age_days": 180.0,
    "amount": 450.0,
    "sender_amount_ratio": 1.0,
    "is_new_receiver": 0.0,
    "sender_tx_count_1h": 1.0,
    "device_changed": 0.0,
    "receiver_distinct_senders_24h": 1.0,
    "receiver_distinct_receivers_24h": 1.0,
    "pass_through_ratio_30m": 0.0,
    "minutes_since_last_inflow": 9999.0
}


def generate_counterfactual(feature_vector, amount, lgb_model, calibrator, feature_names):
    """
    Counterfactual Explanation:
    For flagged payments only, changes each of the top 3 contributing facts one at a time
    to the typical value seen in ALLOW payments, re-scores, and reports the smallest single
    change that flips the decision to ALLOW.
    Visible to fraud operations officers only.
    """
    vec_2d = feature_vector.reshape(1, -1)
    try:
        contribs = lgb_model.booster_.predict(vec_2d, pred_contrib=True)[0, :-1]
    except Exception:
        return ""

    sorted_indices = np.argsort(-contribs)
    top_indices = [idx for idx in sorted_indices if contribs[idx] > 0]
    if not top_indices:
        top_indices = list(sorted_indices[:3])
    else:
        top_indices = top_indices[:3]

    for f_idx in top_indices:
        feat_name = feature_names[f_idx]
        test_vec = feature_vector.copy()
        test_vec[f_idx] = TYPICAL_ALLOW_VALUES.get(feat_name, 0.0)

        test_amt = amount
        if feat_name == "amount":
            test_amt = min(15000.0, max(1000.0, amount * 0.25))

        test_vec_2d = test_vec.reshape(1, -1)
        try:
            raw_p = lgb_model.predict_proba(test_vec_2d)[:, 1]
            if calibrator is not None:
                cal_p = float(np.clip(calibrator.predict(raw_p)[0], 0.0, 1.0))
            else:
                cal_p = float(raw_p[0])
        except Exception:
            continue

        exp_loss = cal_p * test_amt * (1.0 - RECOVERY_RATE)
        fric = STEP_UP_COST + (1.0 - cal_p) * ABANDON_PROB * test_amt

        new_decision = "HOLD" if exp_loss >= HOLD_THRESHOLD else ("STEP-UP" if exp_loss >= fric else "ALLOW")

        if new_decision == "ALLOW":
            if feat_name == "sender_age_days":
                return "would be allowed if sender account age were above 30 days"
            elif feat_name == "receiver_age_days":
                return "would be allowed if receiver account age were above 30 days"
            elif feat_name == "is_new_receiver":
                return "would be allowed if recipient were a previously verified contact"
            elif feat_name == "device_changed":
                return "would be allowed if initiated from user's primary registered device"
            elif feat_name == "amount":
                return f"would be allowed if transfer amount were under Rs {int(max(1000, test_amt)):,}"
            elif feat_name == "pass_through_ratio_30m":
                return "would be allowed if recipient did not rapidly forward funds within 30 minutes"
            elif feat_name == "minutes_since_last_inflow":
                return "would be allowed if settlement turnaround were standard"
            elif feat_name == "sender_tx_count_1h":
                return "would be allowed if sender velocity were within normal hourly limits"
            elif feat_name == "sender_amount_ratio":
                return "would be allowed if amount matched sender's historical spending profile"
            elif feat_name == "receiver_distinct_senders_24h":
                return "would be allowed if recipient inbound velocity were standard"
            else:
                return f"would be allowed if {feat_name} were at normal baseline"

    return "would require both familiar recipient status and lower transfer amount to allow"


def compute_cost_policy(p, amount, is_fraud=None, recovery_rate=RECOVERY_RATE,
                        step_up_cost=STEP_UP_COST, abandon_rate=ABANDON_PROB,
                        hold_threshold=HOLD_THRESHOLD):
    """
    ONE shared function used by the decision engine, the totals and the sliders.
    Formulas:
      expected_loss = p * amount * (1 - recovery_rate)
      friction_estimate = step_up_cost + (1 - p) * abandon_rate * amount
      decision: HOLD if expected_loss > hold_threshold;
                else STEP-UP if expected_loss > friction_estimate;
                else ALLOW.
    Totals over replay or test payments (when is_fraud is provided):
      rupees_prevented = sum over flagged payments with is_fraud = 1 of amount * (1 - recovery_rate)
      friction_incurred = sum over all flagged payments of step_up_cost
                          + sum over flagged payments with is_fraud = 0 of abandon_rate * amount
      multiplier = rupees_prevented / friction_incurred (show "-" if friction is 0)
    """
    p_arr = np.asarray(p, dtype=np.float64)
    amt_arr = np.asarray(amount, dtype=np.float64)
    is_scalar = (p_arr.ndim == 0 and amt_arr.ndim == 0)

    expected_loss = p_arr * amt_arr * (1.0 - recovery_rate)
    friction_estimate = step_up_cost + (1.0 - p_arr) * abandon_rate * amt_arr

    is_hold = expected_loss > hold_threshold
    is_stepup = (~is_hold) & (expected_loss > friction_estimate)
    is_allow = (~is_hold) & (~is_stepup)
    is_flagged = is_hold | is_stepup

    if is_scalar:
        if is_hold:
            decision = "HOLD"
        elif is_stepup:
            decision = "STEP-UP"
        else:
            decision = "ALLOW"
        return decision, float(expected_loss), float(friction_estimate)

    decisions = np.where(is_hold, "HOLD", np.where(is_stepup, "STEP-UP", "ALLOW"))
    counts = {
        "ALLOW": int(np.sum(is_allow)),
        "STEP-UP": int(np.sum(is_stepup)),
        "HOLD": int(np.sum(is_hold))
    }
    total_tx = max(len(p_arr), 1)
    allow_pct = round((counts["ALLOW"] / total_tx) * 100.0, 1)
    step_up_pct = round((counts["STEP-UP"] / total_tx) * 100.0, 1)
    hold_pct = round((counts["HOLD"] / total_tx) * 100.0, 1)
    flagged_share_pct = round((np.sum(is_flagged) / total_tx) * 100.0, 2)

    result = {
        "decisions": decisions,
        "expected_loss": expected_loss,
        "friction_estimate": friction_estimate,
        "counts": counts,
        "allow_pct": allow_pct,
        "step_up_pct": step_up_pct,
        "hold_pct": hold_pct,
        "flagged_share_pct": flagged_share_pct,
        "is_flagged": is_flagged
    }

    if is_fraud is not None:
        fraud_arr = np.asarray(is_fraud, dtype=np.int64)
        good_mask = (fraud_arr == 0)
        fraud_mask = (fraud_arr == 1)

        rupees_prevented = float(np.sum(amt_arr[is_flagged & fraud_mask] * (1.0 - recovery_rate)))
        friction_incurred = float(np.sum(step_up_cost * is_flagged) + np.sum(abandon_rate * amt_arr[is_flagged & good_mask]))

        if friction_incurred > 0:
            multiplier = round(rupees_prevented / friction_incurred, 1)
        else:
            multiplier = "-"

        total_good = max(int(np.sum(good_mask)), 1)
        good_flagged = int(np.sum(is_flagged & good_mask))
        false_alarm_rate_pct = round((good_flagged / total_good) * 100.0, 2)
        net_benefit = rupees_prevented - friction_incurred

        result.update({
            "rupees_prevented": round(rupees_prevented, 2),
            "rupees_friction": round(friction_incurred, 2),
            "friction_incurred": round(friction_incurred, 2),
            "net_benefit": round(net_benefit, 2),
            "multiplier": multiplier,
            "false_alarm_rate_pct": false_alarm_rate_pct
        })

    return result


def evaluate_decision_and_reasons(amount, fraud_prob, feature_vector, lgb_model, feature_names, calibrator=None):
    """
    REUSABLE FUNCTION:
    Computes expected loss, friction, decision (ALLOW, STEP-UP, HOLD),
    officer_reasons (up to 3 plain-English facts), customer_message (safe, non-leaking),
    and counterfactual ('what would change this decision').
    """
    decision, expected_loss, friction = compute_cost_policy(fraud_prob, amount)
        
    officer_reasons = []
    customer_message = generate_customer_message(decision)
    counterfactual = ""

    if decision in ["STEP-UP", "HOLD"]:
        # Compute TreeSHAP contributions for this transaction
        vec_2d = feature_vector.reshape(1, -1)
        try:
            contribs = lgb_model.booster_.predict(vec_2d, pred_contrib=True)[0, :-1]
        except Exception:
            contribs = np.zeros(len(feature_names))

        sorted_indices = np.argsort(-contribs)
        
        for idx in sorted_indices:
            if len(officer_reasons) >= 3:
                break
            feat_name = feature_names[idx]
            val = feature_vector[idx]
            if contribs[idx] > 0.0 or len(officer_reasons) == 0:
                officer_reasons.append(format_feature_reason(feat_name, val))
                
        if not officer_reasons:
            officer_reasons.append(f"Elevated risk score ({fraud_prob*100:.1f}%) exceeding safety policy")

        # Compute Counterfactual
        counterfactual = generate_counterfactual(feature_vector, amount, lgb_model, calibrator, feature_names)

    return decision, expected_loss, friction, officer_reasons, customer_message, counterfactual


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
    
    # 2. Shared policy decisioning
    policy_res = compute_cost_policy(
        hybrid_scores, amounts, is_fraud=y_test,
        recovery_rate=RECOVERY_RATE,
        step_up_cost=STEP_UP_COST,
        abandon_rate=ABANDON_PROB,
        hold_threshold=HOLD_THRESHOLD
    )
    decisions = list(policy_res["decisions"])
    rupees_prevented = policy_res["rupees_prevented"]
    rupees_friction = policy_res["rupees_friction"]
    flagged_indices = np.where(policy_res["is_flagged"])[0].tolist()
                    
    test_df["score"] = hybrid_scores
    test_df["decision"] = decisions
    
    allow_count = policy_res["counts"]["ALLOW"]
    step_up_count = policy_res["counts"]["STEP-UP"]
    hold_count = policy_res["counts"]["HOLD"]
    
    print("=" * 70)
    print(" BUSINESS DECISION SUMMARY (TEST PERIOD)")
    print("=" * 70)
    print(f"Total Evaluated : {n_test:,}")
    print(f"ALLOW Decisions : {allow_count:,} ({policy_res['allow_pct']}%)")
    print(f"STEP-UP (Verify): {step_up_count:,} ({policy_res['step_up_pct']}%)")
    print(f"HOLD (Blocked)  : {hold_count:,} ({policy_res['hold_pct']}%)")
    print("-" * 70)
    print(f"Rupees of Fraud Prevented : Rs {rupees_prevented:,.2f}")
    print(f"Rupees of Friction Caused : Rs {rupees_friction:,.2f}")
    roi = policy_res["multiplier"]
    roi_str = f"{roi}x" if roi != "-" else "-"
    print(f"Net Protection Multiplier : {roi_str} (saved Rs {roi_str} for every Rs 1 of friction)")
    print("=" * 70 + "\n")
    
    # 3. Print 10 Flagged Payments with their Plain-English Reasons
    print("=" * 70)
    print(" 10 FLAGGED PAYMENTS WITH REAL-TIME EXPLAINABILITY REASONS")
    print("=" * 70)
    
    sample_show = flagged_indices[:10]
    for idx_num, i in enumerate(sample_show):
        row = test_df.iloc[i]
        dec, exp_loss, fric, officer_reasons, cust_msg, cf = evaluate_decision_and_reasons(
            amounts[i], hybrid_scores[i], X_test[i], lgb_model, feature_cols, calibrator=calibrator
        )
        fraud_label = "TRUE FRAUD" if row['is_fraud'] == 1 else "GOOD CUSTOMER"
        print(f"[{idx_num + 1}] Txn: {row['txn_id']} | Decision: {row['decision']} | Score: {row['score']:.3f} | Rs {row['amount']:,.2f} | ({fraud_label})")
        print(f"    Sender: {row['sender']} -> Receiver: {row['receiver']}")
        print(f"    Customer Notice: \"{cust_msg}\"")
        print("    Officer Reasons:")
        for r in officer_reasons:
            print(f"      • {r}")
        if cf:
            print(f"    Counterfactual (Officers Only): {cf}")
        print()
    print("=" * 70)
    print("[SUCCESS] Step 5 complete: Decision engine & explainability ready.")


if __name__ == "__main__":
    main()

