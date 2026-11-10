"""
StreamShield - Step 3: Stateful Real-Time Feature Engine
Maintains in-memory account state (O(1) lookups) using pure Python dict and deque.
Computes 10 behavioral & network signals per transaction using strictly past events.

Exposes ONE reusable function: compute_features_and_update() used both for
batch feature engineering and live streaming replay.
"""

import os
import time
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# ==========================================
# TIME WINDOW CONSTANTS (in seconds)
# ==========================================
ONE_HOUR = 3600
THIRTY_MINUTES = 1800
TWENTY_FOUR_HOURS = 86400
SECONDS_PER_DAY = 86400.0


class AccountMemoryState:
    """
    Lightweight, high-performance in-memory state store for UPI accounts.
    Maintains O(1) rolling time-window statistics without database overhead.
    """
    def __init__(self):
        # Account first seen timestamps: acc -> timestamp
        self.first_seen = {}
        
        # Primary device tracking: acc -> device_id
        self.primary_device = {}
        
        # Historical recipient sets: sender -> set(receivers)
        self.known_recipients = defaultdict(set)
        
        # Sender historical spending stats: sender -> [total_amount, count]
        self.sender_amount_stats = defaultdict(lambda: [0.0, 0])
        
        # Sender velocity: sender -> deque of timestamps in last 1 hour
        self.sender_outbound_1h = defaultdict(deque)
        
        # Receiver incoming senders in last 24h: receiver -> deque of (time, sender)
        self.receiver_inbound_24h = defaultdict(deque)
        self.receiver_sender_counts = defaultdict(lambda: defaultdict(int))
        self.receiver_distinct_senders = defaultdict(int)
        
        # Receiver outbound recipients in last 24h: account -> deque of (time, receiver)
        self.receiver_outbound_24h = defaultdict(deque)
        self.receiver_recipient_counts = defaultdict(lambda: defaultdict(int))
        self.receiver_distinct_recipients = defaultdict(int)
        
        # 30-minute cash flow for pass-through ratio: account -> deque of (time, amount)
        self.inflow_30m = defaultdict(deque)
        self.inflow_30m_sum = defaultdict(float)
        
        self.outflow_30m = defaultdict(deque)
        self.outflow_30m_sum = defaultdict(float)
        
        # Last incoming transfer timestamp: receiver -> timestamp
        self.last_inflow_time = {}


def compute_features_and_update(tx, state):
    """
    REUSABLE FUNCTION:
    Computes feature vector for a single transaction using ONLY past information,
    then updates the state. Used both in batch dataset creation and live streaming.
    
    Parameters:
        tx: dict with keys ['time', 'sender', 'receiver', 'amount', 'device_id']
        state: AccountMemoryState instance
        
    Returns:
        features: dict containing computed numerical facts
    """
    t = tx["time"]
    sender = tx["sender"]
    receiver = tx["receiver"]
    amt = float(tx["amount"])
    device = tx["device_id"]
    
    # -----------------------------------------------------------
    # 1. Account Ages (in days)
    # -----------------------------------------------------------
    if sender not in state.first_seen:
        state.first_seen[sender] = t
    sender_age_days = (t - state.first_seen[sender]) / SECONDS_PER_DAY
    
    if receiver not in state.first_seen:
        state.first_seen[receiver] = t
    receiver_age_days = (t - state.first_seen[receiver]) / SECONDS_PER_DAY
    
    # -----------------------------------------------------------
    # 2. Sender Amount Compared to Usual
    # -----------------------------------------------------------
    sender_stats = state.sender_amount_stats[sender]
    if sender_stats[1] > 0:
        usual_amt = sender_stats[0] / sender_stats[1]
        sender_amount_ratio = amt / max(usual_amt, 10.0)
    else:
        # First time sending money; baseline ratio is 1.0
        sender_amount_ratio = 1.0
        
    # -----------------------------------------------------------
    # 3. New Receiver for Sender (0 or 1)
    # -----------------------------------------------------------
    is_new_receiver = 1 if receiver not in state.known_recipients[sender] else 0
    
    # -----------------------------------------------------------
    # 4. Sender Velocity (Payments in last 1 hour)
    # -----------------------------------------------------------
    outbound_1h = state.sender_outbound_1h[sender]
    cutoff_1h = t - ONE_HOUR
    while outbound_1h and outbound_1h[0] < cutoff_1h:
        outbound_1h.popleft()
    sender_tx_count_1h = len(outbound_1h)
    
    # -----------------------------------------------------------
    # 5. Device Changed (0 or 1)
    # -----------------------------------------------------------
    if sender not in state.primary_device:
        state.primary_device[sender] = device
        device_changed = 0
    else:
        device_changed = 1 if state.primary_device[sender] != device else 0
        
    # -----------------------------------------------------------
    # 6. Receiver: Distinct senders in last 24 hours
    # -----------------------------------------------------------
    inbound_24h = state.receiver_inbound_24h[receiver]
    s_counts = state.receiver_sender_counts[receiver]
    cutoff_24h = t - TWENTY_FOUR_HOURS
    
    while inbound_24h and inbound_24h[0][0] < cutoff_24h:
        _, old_sender = inbound_24h.popleft()
        s_counts[old_sender] -= 1
        if s_counts[old_sender] == 0:
            del s_counts[old_sender]
            state.receiver_distinct_senders[receiver] -= 1
            
    receiver_distinct_senders_24h = state.receiver_distinct_senders[receiver]
    
    # -----------------------------------------------------------
    # 7. Receiver: Distinct receivers it paid in last 24 hours
    # -----------------------------------------------------------
    rec_outbound_24h = state.receiver_outbound_24h[receiver]
    r_counts = state.receiver_recipient_counts[receiver]
    
    while rec_outbound_24h and rec_outbound_24h[0][0] < cutoff_24h:
        _, old_rec = rec_outbound_24h.popleft()
        r_counts[old_rec] -= 1
        if r_counts[old_rec] == 0:
            del r_counts[old_rec]
            state.receiver_distinct_recipients[receiver] -= 1
            
    receiver_distinct_receivers_24h = state.receiver_distinct_recipients[receiver]
    
    # -----------------------------------------------------------
    # 8. Pass-Through Ratio (30-minute cash flow for receiver)
    #    Money sent out in last 30 mins / Money received in last 30 mins
    # -----------------------------------------------------------
    in_30m = state.inflow_30m[receiver]
    cutoff_30m = t - THIRTY_MINUTES
    
    while in_30m and in_30m[0][0] < cutoff_30m:
        _, old_amt = in_30m.popleft()
        state.inflow_30m_sum[receiver] -= old_amt
        
    out_30m = state.outflow_30m[receiver]
    while out_30m and out_30m[0][0] < cutoff_30m:
        _, old_amt = out_30m.popleft()
        state.outflow_30m_sum[receiver] -= old_amt
        
    total_in = state.inflow_30m_sum[receiver]
    total_out = state.outflow_30m_sum[receiver]
    
    if total_in > 10.0:
        pass_through_ratio_30m = round(min(5.0, total_out / total_in), 4)
    else:
        pass_through_ratio_30m = 0.0
        
    # -----------------------------------------------------------
    # 9. Minutes since receiver's last incoming payment
    # -----------------------------------------------------------
    if receiver in state.last_inflow_time:
        minutes_since_last_inflow = (t - state.last_inflow_time[receiver]) / 60.0
    else:
        # No recorded prior inbound payment
        minutes_since_last_inflow = 9999.0
        
    # ===========================================================
    # STATE MUTATION: Update state WITH the current transaction
    # =================================----------------==========
    # Update sender stats
    sender_stats[0] += amt
    sender_stats[1] += 1
    state.known_recipients[sender].add(receiver)
    outbound_1h.append(t)
    
    # Update receiver inbound tracking
    inbound_24h.append((t, sender))
    if s_counts[sender] == 0:
        state.receiver_distinct_senders[receiver] += 1
    s_counts[sender] += 1
    
    # Update sender outbound tracking for 24h distinct recipients
    sender_out_24h = state.receiver_outbound_24h[sender]
    sender_r_counts = state.receiver_recipient_counts[sender]
    sender_out_24h.append((t, receiver))
    if sender_r_counts[receiver] == 0:
        state.receiver_distinct_recipients[sender] += 1
    sender_r_counts[receiver] += 1
    
    # Update 30m cash flows
    in_30m.append((t, amt))
    state.inflow_30m_sum[receiver] += amt
    
    sender_out_30m = state.outflow_30m[sender]
    sender_out_30m.append((t, amt))
    state.outflow_30m_sum[sender] += amt
    
    # Update last inflow timestamp
    state.last_inflow_time[receiver] = t
    
    return {
        "sender_age_days": round(sender_age_days, 3),
        "receiver_age_days": round(receiver_age_days, 3),
        "sender_amount_ratio": round(sender_amount_ratio, 3),
        "is_new_receiver": is_new_receiver,
        "sender_tx_count_1h": sender_tx_count_1h,
        "device_changed": device_changed,
        "receiver_distinct_senders_24h": receiver_distinct_senders_24h,
        "receiver_distinct_receivers_24h": receiver_distinct_receivers_24h,
        "pass_through_ratio_30m": pass_through_ratio_30m,
        "minutes_since_last_inflow": round(minutes_since_last_inflow, 2)
    }


def process_dataset(input_csv, output_csv):
    """
    Processes all payments in time order and writes the full feature table.
    """
    print(f"\nProcessing {input_csv} -> {output_csv}...")
    start_time = time.time()
    
    df = pd.read_csv(input_csv)
    n_rows = len(df)
    state = AccountMemoryState()
    
    feature_records = []
    
    # Convert dataframe columns to list of dicts for maximum iteration speed
    records = df[["time", "sender", "receiver", "amount", "device_id"]].to_dict(orient="records")
    
    for i, tx in enumerate(records):
        feats = compute_features_and_update(tx, state)
        feature_records.append(feats)
        
    feat_df = pd.DataFrame(feature_records)
    
    # Combine original identity columns + computed features + labels
    result_df = pd.concat([
        df[["txn_id", "time", "sender", "receiver", "amount", "device_id", "channel"]],
        feat_df,
        df[["is_fraud", "ring_id"]]
    ], axis=1)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    speed = n_rows / max(elapsed, 0.001)
    print(f"Processed {n_rows:,} transactions in {elapsed:.2f} seconds ({speed:,.0f} tx/sec).")
    print(f"Saved to {output_csv}")
    
    return result_df


def show_collector_sample(df, round_label):
    """
    Finds and displays 5 sample rows for a mule ring collector.
    """
    ring_rows = df[df["ring_id"].str.len() > 0]
    if len(ring_rows) == 0:
        return
        
    # A collector is a receiver in ring payments that receives from multiple accounts
    sample_ring = ring_rows["ring_id"].iloc[0]
    sub = df[df["ring_id"] == sample_ring]
    
    # Find the account that appears as receiver most often in this ring
    collector_acc = sub["receiver"].value_counts().index[0]
    collector_rows = sub[(sub["receiver"] == collector_acc) | (sub["sender"] == collector_acc)].head(5)
    
    display_cols = [
        "txn_id", "time", "sender", "receiver", "amount",
        "receiver_distinct_senders_24h", "pass_through_ratio_30m",
        "sender_amount_ratio", "is_fraud", "ring_id"
    ]
    print(f"\nExample 5 feature rows for Mule Collector ({collector_acc}) in {sample_ring} ({round_label}):")
    print(collector_rows[display_cols].to_string(index=False))


def main():
    r1_in = os.path.join("data", "round1.csv")
    r1_out = os.path.join("data", "features_round1.csv")
    r2_in = os.path.join("data", "round2.csv")
    r2_out = os.path.join("data", "features_round2.csv")
    
    df1 = process_dataset(r1_in, r1_out)
    show_collector_sample(df1, "Round 1")
    
    df2 = process_dataset(r2_in, r2_out)
    show_collector_sample(df2, "Round 2")
    
    print("\n[SUCCESS] Step 3 complete: Feature datasets ready for machine learning.")


if __name__ == "__main__":
    main()
