"""
StreamShield - Step 1: Synthetic Data Generator
Generates realistic UPI transactions for Round 1 and Round 2 with planted mule rings and scams.
"""

import os
import random
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION SETTINGS (Adjust as needed)
# ==========================================
RANDOM_SEED = 42

# Accounts & Activity
NUM_ACCOUNTS = 35000          # 35,000 unique UPI accounts
NEW_ACCOUNT_RATIO = 0.15     # 15% of accounts are brand-new (appear late)
TOTAL_DAYS = 14
TOTAL_SECONDS = TOTAL_DAYS * 24 * 3600  # 1,209,600 seconds
NUM_NORMAL_PAYMENTS = 600000 # 600,000 normal baseline payments (1.2M+ total)

# Fraud Settings
NUM_RINGS = 30               # 30 mule rings per round
NUM_SCAMS = 220              # 220 social engineering scam payments (<0.1% fraud rate)

CHANNELS = ["UPI_APP", "QR_CODE", "INTENT", "COLLECT_REQUEST", "P2P"]
CHANNEL_WEIGHTS = [0.45, 0.30, 0.12, 0.08, 0.05]

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_account_universe(num_accounts, new_account_ratio, total_seconds):
    """
    Creates account metadata: creation time, primary device, and favorite contacts.
    """
    accounts = [f"acc_{i:05d}" for i in range(num_accounts)]
    num_new = int(num_accounts * new_account_ratio)
    
    # 85% established accounts (created before simulation start or day 0)
    # 15% brand-new accounts (created after day 7 of 14)
    creation_times = {}
    midpoint = total_seconds * 0.5
    for i, acc in enumerate(accounts):
        if i < num_new:
            # Brand-new account created in days 7-13
            creation_times[acc] = random.uniform(midpoint, total_seconds - 3600)
        else:
            # Established account created before or at day 0
            creation_times[acc] = 0.0

    devices = {acc: f"dev_{acc}" for acc in accounts}
    
    # Each account has 2-5 favorite receivers to simulate realistic repeated transfers
    favorites = {}
    for acc in accounts:
        k = random.randint(2, 5)
        pool = random.sample(accounts, k + 1)
        favorites[acc] = [x for x in pool if x != acc][:k]
        
    return accounts, creation_times, devices, favorites


def generate_normal_payments(accounts, creation_times, devices, favorites, num_payments, total_seconds):
    """
    Generates normal, legitimate UPI payments spread across 14 days.
    """
    rows = []
    
    # Pre-sample timestamps with daytime activity weighting
    raw_times = np.random.uniform(0, total_seconds, size=num_payments)
    # Add diurnal weight (less activity between midnight and 6 AM)
    hour_of_day = (raw_times % 86400) / 3600.0
    night_mask = (hour_of_day < 6)
    # Shift night transactions into active daytime hours
    raw_times[night_mask] += np.random.uniform(6 * 3600, 16 * 3600, size=np.sum(night_mask))
    raw_times = np.clip(raw_times, 0, total_seconds - 1)
    raw_times.sort()
    
    # Sample payment amounts: 3-tier realistic UPI mixture distribution
    # Tier 1 (65%): Micro-payments (tea, grocery, auto) ₹20 - ₹1,500
    # Tier 2 (25%): Mid-range transfers & shopping ₹2,000 - ₹18,000
    # Tier 3 (10%): High-value legitimate transfers (rent, fees, electronics) ₹20,000 - ₹1,20,000 (Lakhs!)
    tier_choices = np.random.choice([1, 2, 3], size=num_payments, p=[0.65, 0.25, 0.10])
    
    t1_amounts = np.random.lognormal(mean=5.2, sigma=0.8, size=num_payments)
    t1_amounts = np.clip(t1_amounts, 20.0, 1500.0)
    
    t2_amounts = np.random.lognormal(mean=8.5, sigma=0.6, size=num_payments)
    t2_amounts = np.clip(t2_amounts, 2000.0, 18000.0)
    
    t3_amounts = np.random.uniform(20000.0, 120000.0, size=num_payments)
    
    amounts = np.where(tier_choices == 1, t1_amounts, np.where(tier_choices == 2, t2_amounts, t3_amounts))
    amounts = np.round(amounts, 2)
    
    channels = np.random.choice(CHANNELS, size=num_payments, p=CHANNEL_WEIGHTS)
    
    for i in range(num_payments):
        t = float(raw_times[i])
        sender = random.choice(accounts)
        
        # If sender is new and transaction is before account birth, push transaction to after birth
        if t < creation_times[sender]:
            t = creation_times[sender] + random.uniform(60, 3600)
            if t >= total_seconds:
                t = total_seconds - random.uniform(10, 600)
                
        # 75% of the time, pay a favorite; 25% pay a random account
        if random.random() < 0.75 and sender in favorites and favorites[sender]:
            receiver = random.choice(favorites[sender])
        else:
            receiver = random.choice(accounts)
            while receiver == sender:
                receiver = random.choice(accounts)
                
        # 98% of payments use the primary registered device; 2% use an alternate device
        dev = devices[sender] if random.random() < 0.98 else f"dev_alt_{sender}"
        
        rows.append({
            "txn_id": f"txn_{i:07d}",
            "time": int(t),
            "sender": sender,
            "receiver": receiver,
            "amount": float(amounts[i]),
            "device_id": dev,
            "channel": channels[i],
            "is_fraud": 0,
            "ring_id": ""
        })
        
    return rows


def generate_round1_fraud(accounts, devices, total_seconds, txn_offset):
    """
    Round 1 Fraud:
    - 20 Mule Rings: 6-12 senders pay 1 collector within 10 min;
      collector forwards >= 90% onward to 1-2 exits within 5 min.
    - 150 Scam Payments: large first-time payments to previously unseen receivers.
    """
    fraud_rows = []
    current_txn_id = txn_offset
    
    # 1. Mule Rings
    ring_start_times = np.linspace(86400, total_seconds - 7200, NUM_RINGS)
    
    for r_idx in range(NUM_RINGS):
        ring_id = f"ring_{r_idx + 1:02d}"
        t_base = int(ring_start_times[r_idx]) + random.randint(0, 1800)
        
        num_feeders = random.randint(6, 12)
        # Choose unique accounts for the ring
        ring_accounts = random.sample(accounts, num_feeders + 3)
        collector = ring_accounts[0]
        feeders = ring_accounts[1:num_feeders + 1]
        exits = ring_accounts[num_feeders + 1:]
        
        total_inflow = 0.0
        feeder_times = []
        
        # Step A: Feeders pay collector within 10 minutes (600 seconds)
        for feeder in feeders:
            feeder_time = t_base + random.randint(10, 580)
            feeder_times.append(feeder_time)
            amount = round(random.uniform(15000, 45000), 2)
            total_inflow += amount
            
            fraud_rows.append({
                "txn_id": f"txn_{current_txn_id:07d}",
                "time": feeder_time,
                "sender": feeder,
                "receiver": collector,
                "amount": amount,
                "device_id": devices[feeder],
                "channel": "UPI_APP",
                "is_fraud": 1,
                "ring_id": ring_id
            })
            current_txn_id += 1
            
        # Step B: Collector forwards >= 90% onward to 1-2 exit accounts within 5 minutes (300s)
        last_inflow_time = max(feeder_times)
        payout_ratio = random.uniform(0.90, 0.98)
        total_payout = round(total_inflow * payout_ratio, 2)
        
        num_exits = min(len(exits), random.choice([1, 2]))
        payout_chunks = [round(total_payout / num_exits, 2)] * num_exits
        
        for k in range(num_exits):
            out_time = last_inflow_time + random.randint(30, 290)
            fraud_rows.append({
                "txn_id": f"txn_{current_txn_id:07d}",
                "time": out_time,
                "sender": collector,
                "receiver": exits[k],
                "amount": payout_chunks[k],
                "device_id": devices[collector],
                "channel": "UPI_APP",
                "is_fraud": 1,
                "ring_id": ring_id
            })
            current_txn_id += 1

    # 2. Scam Payments (Large first-time transfers in thousands and lakhs)
    scam_times = np.linspace(3600, total_seconds - 3600, NUM_SCAMS)
    for s_idx in range(NUM_SCAMS):
        t_scam = int(scam_times[s_idx]) + random.randint(-600, 600)
        victim, scammer = random.sample(accounts, 2)
        scam_amount = round(random.uniform(45000, 185000), 2)
        
        fraud_rows.append({
            "txn_id": f"txn_{current_txn_id:07d}",
            "time": t_scam,
            "sender": victim,
            "receiver": scammer,
            "amount": scam_amount,
            "device_id": devices[victim],
            "channel": "INTENT",
            "is_fraud": 1,
            "ring_id": ""
        })
        current_txn_id += 1
        
    return fraud_rows, current_txn_id


def generate_round2_fraud(accounts, devices, total_seconds, txn_offset):
    """
    Round 2 Fraud (Adversarial Concept Drift):
    - Collector waits 30 to 120 minutes before forwarding.
    - Money is split into several smaller structured payments (smurfing).
    - Some rings use multi-hop chains (A -> B -> C -> D).
    - Fraud accounts use new device IDs (device hopping).
    """
    fraud_rows = []
    current_txn_id = txn_offset
    ring_start_times = np.linspace(86400, total_seconds - 14400, NUM_RINGS)
    
    for r_idx in range(NUM_RINGS):
        ring_id = f"ring_r2_{r_idx + 1:02d}"
        t_base = int(ring_start_times[r_idx]) + random.randint(0, 1800)
        
        # Decide if this ring uses a linear multi-hop chain (A -> B -> C -> D)
        use_chain = (r_idx % 4 == 0)
        
        if use_chain:
            # 4-hop chain
            chain_nodes = random.sample(accounts, 4)
            chain_amount = round(random.uniform(75000, 220000), 2)
            hop_time = t_base
            
            for hop in range(3):
                hop_time += random.randint(600, 2400) # 10 to 40 mins per hop
                src_acc = chain_nodes[hop]
                dst_acc = chain_nodes[hop + 1]
                # New device ID used by fraud accounts
                new_dev = f"dev_drift_{src_acc}_{r_idx}"
                
                fraud_rows.append({
                    "txn_id": f"txn_{current_txn_id:07d}",
                    "time": hop_time,
                    "sender": src_acc,
                    "receiver": dst_acc,
                    "amount": round(chain_amount * (0.95 ** hop), 2),
                    "device_id": new_dev,
                    "channel": "UPI_APP",
                    "is_fraud": 1,
                    "ring_id": ring_id
                })
                current_txn_id += 1
        else:
            # Delayed forwarding + Smurfing (split into 4 to 8 smaller transfers)
            num_feeders = random.randint(6, 12)
            ring_accounts = random.sample(accounts, num_feeders + 5)
            collector = ring_accounts[0]
            feeders = ring_accounts[1:num_feeders + 1]
            exits = ring_accounts[num_feeders + 1:]
            
            total_inflow = 0.0
            feeder_times = []
            
            for feeder in feeders:
                feeder_time = t_base + random.randint(60, 1800)
                feeder_times.append(feeder_time)
                amount = round(random.uniform(12000, 38000), 2)
                total_inflow += amount
                new_dev = f"dev_drift_{feeder}_{r_idx}"
                
                fraud_rows.append({
                    "txn_id": f"txn_{current_txn_id:07d}",
                    "time": feeder_time,
                    "sender": feeder,
                    "receiver": collector,
                    "amount": amount,
                    "device_id": new_dev,
                    "channel": "UPI_APP",
                    "is_fraud": 1,
                    "ring_id": ring_id
                })
                current_txn_id += 1
                
            last_inflow = max(feeder_times)
            # Delayed forwarding: waits 30 to 120 minutes (1,800 to 7,200 seconds)
            forward_start = last_inflow + random.randint(1800, 7200)
            
            # Split money into 4-8 smaller structured transfers
            num_splits = random.randint(4, 8)
            payout_total = total_inflow * random.uniform(0.90, 0.97)
            chunk_size = round(payout_total / num_splits, 2)
            
            collector_new_dev = f"dev_drift_{collector}_{r_idx}"
            for s in range(num_splits):
                out_time = forward_start + (s * random.randint(120, 600))
                dst = random.choice(exits)
                fraud_rows.append({
                    "txn_id": f"txn_{current_txn_id:07d}",
                    "time": out_time,
                    "sender": collector,
                    "receiver": dst,
                    "amount": chunk_size,
                    "device_id": collector_new_dev,
                    "channel": "UPI_APP",
                    "is_fraud": 1,
                    "ring_id": ring_id
                })
                current_txn_id += 1
                
    # Scam Payments for Round 2 in thousands and lakhs
    scam_times = np.linspace(3600, total_seconds - 3600, NUM_SCAMS)
    for s_idx in range(NUM_SCAMS):
        t_scam = int(scam_times[s_idx]) + random.randint(-600, 600)
        victim, scammer = random.sample(accounts, 2)
        scam_amount = round(random.uniform(40000, 160000), 2)
        fraud_rows.append({
            "txn_id": f"txn_{current_txn_id:07d}",
            "time": t_scam,
            "sender": victim,
            "receiver": scammer,
            "amount": scam_amount,
            "device_id": f"dev_alt_{victim}",
            "channel": "INTENT",
            "is_fraud": 1,
            "ring_id": ""
        })
        current_txn_id += 1
        
    return fraud_rows, current_txn_id


def save_dataset(normal_rows, fraud_rows, output_path):
    """
    Combines normal and fraud payments, sorts chronologically, and saves to CSV.
    """
    all_rows = normal_rows + fraud_rows
    df = pd.DataFrame(all_rows)
    df = df.sort_values(by="time").reset_index(drop=True)
    df["txn_id"] = [f"txn_{i:07d}" for i in range(len(df))]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def print_summary(df, round_name):
    """
    Prints key statistics and 5 example rows of one mule ring.
    """
    num_rows = len(df)
    num_fraud = int(df["is_fraud"].sum())
    fraud_pct = (num_fraud / num_rows) * 100.0
    
    unique_rings = [r for r in df["ring_id"].unique() if r != ""]
    num_rings = len(unique_rings)
    
    print("=" * 65)
    print(f" {round_name} DATASET GENERATION SUMMARY")
    print("=" * 65)
    print(f"Total Transactions : {num_rows:,}")
    print(f"Fraud Transactions : {num_fraud:,} ({fraud_pct:.3f}%)")
    print(f"Mule Rings Count   : {num_rings}")
    print("-" * 65)
    
    if unique_rings:
        sample_ring = unique_rings[0]
        print(f"Example 5 rows from {sample_ring}:")
        sample_rows = df[df["ring_id"] == sample_ring].head(5)
        print(sample_rows[["txn_id", "time", "sender", "receiver", "amount", "channel", "ring_id"]].to_string(index=False))
    print("=" * 65 + "\n")


def main():
    print("Initializing StreamShield synthetic UPI universe...")
    accounts, creation_times, devices, favorites = generate_account_universe(
        NUM_ACCOUNTS, NEW_ACCOUNT_RATIO, TOTAL_SECONDS
    )
    
    # ------------------ ROUND 1 ------------------
    print("Generating Round 1 normal payments...")
    r1_normal = generate_normal_payments(
        accounts, creation_times, devices, favorites, NUM_NORMAL_PAYMENTS, TOTAL_SECONDS
    )
    print("Planting Round 1 mule rings and scams...")
    r1_fraud, next_id = generate_round1_fraud(
        accounts, devices, TOTAL_SECONDS, txn_offset=NUM_NORMAL_PAYMENTS
    )
    r1_path = os.path.join("data", "round1.csv")
    df_r1 = save_dataset(r1_normal, r1_fraud, r1_path)
    print_summary(df_r1, "ROUND 1")
    
    # ------------------ ROUND 2 ------------------
    print("Generating Round 2 normal payments...")
    # Re-seed for fresh normal transactions in Round 2
    random.seed(RANDOM_SEED + 100)
    np.random.seed(RANDOM_SEED + 100)
    r2_normal = generate_normal_payments(
        accounts, creation_times, devices, favorites, NUM_NORMAL_PAYMENTS, TOTAL_SECONDS
    )
    print("Planting Round 2 drifted fraud patterns...")
    r2_fraud, _ = generate_round2_fraud(
        accounts, devices, TOTAL_SECONDS, txn_offset=NUM_NORMAL_PAYMENTS
    )
    r2_path = os.path.join("data", "round2.csv")
    df_r2 = save_dataset(r2_normal, r2_fraud, r2_path)
    print_summary(df_r2, "ROUND 2")
    
    print(f"Round 1 saved to: {r1_path}")
    print(f"Round 2 saved to: {r2_path}")


if __name__ == "__main__":
    main()
