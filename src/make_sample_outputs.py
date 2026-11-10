"""
StreamShield - Step 8 Part A: Sample Output Generator
Creates realistic sample files matching the exact schema of production outputs
so the dashboard can be tested independently.
Marks data with is_sample = True.
"""

import os
import json
import random

DECISIONS_PATH = os.path.join("data", "decisions.jsonl")
RINGS_PATH = os.path.join("data", "rings.json")
METRICS_PATH = os.path.join("data", "metrics.json")


def generate_sample_files():
    print("Generating sample dashboard files...")
    os.makedirs("data", exist_ok=True)
    
    random.seed(42)
    
    # 1. Sample Decisions (200 rows)
    decisions = []
    base_time = 1000000
    
    sample_reasons_pool = [
        ["Received money from 8 different senders in the last 24 hours", "Sent out 94 percent of incoming money within 30 minutes", "High payment amount of Rs 22,500.00"],
        ["First-time payment to a receiver this sender never paid before", "Amount is 35.2x higher than this sender's normal payment", "High payment amount of Rs 48,000.00"],
        ["Rapid turnaround: received incoming funds only 2 minutes ago", "First-time payment to a receiver this sender never paid before", "Sender account is only 9 days old"],
        ["Receiver account is only 4 days old", "High payment amount of Rs 18,500.00", "Received money from 6 different senders in the last 24 hours"]
    ]
    
    ring_accounts = ["acc_00101", "acc_00102", "acc_00103", "acc_00104", "acc_00105", "acc_00201", "acc_00202", "acc_00203"]
    normal_accounts = [f"acc_{i:05d}" for i in range(1000, 1050)]
    
    for i in range(200):
        t = base_time + (i * 10)
        txn_id = f"txn_sample_{i:04d}"
        
        # 10% fraud/step-up for demonstration in sample mode
        if i in [15, 32, 48, 65, 84, 110, 135, 155, 178, 192]:
            dec = random.choice(["HOLD", "STEP-UP"])
            score = round(random.uniform(0.80, 0.99), 4)
            sender = random.choice(ring_accounts)
            receiver = random.choice(ring_accounts)
            while receiver == sender:
                receiver = random.choice(ring_accounts)
            amount = round(random.uniform(12000, 65000), 2)
            reasons = random.choice(sample_reasons_pool)
            ring_id = "ring_sample_01" if i < 100 else "ring_sample_02"
            lat = round(random.uniform(15.0, 45.0), 2)
        else:
            dec = "ALLOW"
            score = round(random.uniform(0.001, 0.08), 4)
            sender = random.choice(normal_accounts)
            receiver = random.choice(normal_accounts)
            while receiver == sender:
                receiver = random.choice(normal_accounts)
            amount = round(random.uniform(50, 2500), 2)
            reasons = []
            ring_id = ""
            lat = round(random.uniform(0.01, 12.0), 2)
            
        decisions.append({
            "time": t,
            "txn_id": txn_id,
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "decision": dec,
            "score": score,
            "reasons": reasons,
            "ring_id": ring_id,
            "latency_ms": lat,
            "is_sample": True
        })
        
    with open(DECISIONS_PATH, "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    print(f"  Created {DECISIONS_PATH} (200 sample lines)")
    
    # 2. Sample Rings (4 rings)
    sample_rings = [
        {
            "ring_id": "ring_sample_01",
            "member_accounts": ["acc_00101", "acc_00102", "acc_00103", "acc_00104", "acc_00105"],
            "member_count": 5,
            "total_rupees": 142500.00,
            "time_first_flagged": 1000150,
            "is_sample": True
        },
        {
            "ring_id": "ring_sample_02",
            "member_accounts": ["acc_00201", "acc_00202", "acc_00203", "acc_00204"],
            "member_count": 4,
            "total_rupees": 89400.00,
            "time_first_flagged": 1000840,
            "is_sample": True
        },
        {
            "ring_id": "ring_sample_03",
            "member_accounts": ["acc_00301", "acc_00302", "acc_00303"],
            "member_count": 3,
            "total_rupees": 61200.00,
            "time_first_flagged": 1001350,
            "is_sample": True
        },
        {
            "ring_id": "ring_sample_04",
            "member_accounts": ["acc_00401", "acc_00402", "acc_00403", "acc_00404"],
            "member_count": 4,
            "total_rupees": 115000.00,
            "time_first_flagged": 1001780,
            "is_sample": True
        }
    ]
    with open(RINGS_PATH, "w") as f:
        json.dump(sample_rings, f, indent=2)
    print(f"  Created {RINGS_PATH} (4 sample rings)")
    
    # 3. Sample Metrics
    sample_metrics = {
        "p50": 0.01,
        "p95": 18.25,
        "p99": 28.40,
        "max": 65.10,
        "achieved_tps": 3650.0,
        "pr_auc_rules": 0.5429,
        "pr_auc_model": 0.8966,
        "recall_at_1pct_rules": 80.00,
        "recall_at_1pct_model": 98.33,
        "rupees_prevented": 1229346.16,
        "rupees_friction": 133876.03,
        "is_sample": True
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(sample_metrics, f, indent=2)
    print(f"  Created {METRICS_PATH} (sample metrics)")
    print("\n[SUCCESS] Sample outputs ready for dashboard development.")


if __name__ == "__main__":
    generate_sample_files()
