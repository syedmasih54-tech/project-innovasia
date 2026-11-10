"""
StreamShield - Step 7: Graph-Based Mule Ring Detection
Reads data/decisions.jsonl, extracts high-risk transactions (HOLD or STEP-UP),
and joins accounts into network groups using networkx connected components.

Identifies all clusters with 3 or more accounts as mule rings.
Saves details to data/rings.json and evaluates:
- Ring detection recall (how many real rings we found)
- Average time-to-detection (seconds from ring start to first detection)
"""

import os
import json
from collections import defaultdict
import networkx as nx
import pandas as pd

DECISIONS_PATH = os.path.join("data", "decisions.jsonl")
ROUND1_PATH = os.path.join("data", "round1.csv")
RINGS_OUTPUT = os.path.join("data", "rings.json")


def load_flagged_decisions(path):
    """
    Reads decisions.jsonl and filters for high-score payments (HOLD or STEP-UP).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Decisions file not found at {path}. Run src/replay.py first.")
        
    flagged = []
    total_txns = 0
    all_decisions = []
    
    with open(path, "r") as f:
        for line in f:
            total_txns += 1
            rec = json.loads(line.strip())
            all_decisions.append(rec)
            if rec.get("decision") in ["HOLD", "STEP-UP"]:
                flagged.append(rec)
                
    return flagged, all_decisions, total_txns


def find_mule_rings(flagged_txns):
    """
    Constructs an account transaction graph from flagged payments and finds
    connected components with 3 or more accounts.
    """
    G = nx.Graph()
    
    # Track metadata per edge/transaction
    account_amounts = defaultdict(float)
    account_flag_times = defaultdict(list)
    account_true_rings = defaultdict(set)
    
    for tx in flagged_txns:
        u = tx["sender"]
        v = tx["receiver"]
        amt = float(tx["amount"])
        t = tx["time"]
        rid = tx.get("ring_id", "")
        
        G.add_edge(u, v, amount=amt, time=t)
        account_amounts[u] += amt
        account_amounts[v] += amt
        account_flag_times[u].append(t)
        account_flag_times[v].append(t)
        
        if rid and rid != "nan":
            account_true_rings[u].add(rid)
            account_true_rings[v].add(rid)
            
    # Find connected components (groups of accounts that transacted together)
    components = list(nx.connected_components(G))
    
    detected_rings = []
    ring_counter = 1
    
    for comp in components:
        # Every group with 3 or more accounts is a mule ring
        if len(comp) >= 3:
            members = sorted(list(comp))
            
            # Compute total rupees and earliest flag time across this ring
            sub_g = G.subgraph(comp)
            total_rupees = sum(d["amount"] for _, _, d in sub_g.edges(data=True))
            earliest_flag = min(d["time"] for _, _, d in sub_g.edges(data=True))
            
            # Determine overlapping true ring if present
            true_labels = set()
            for m in comp:
                true_labels.update(account_true_rings[m])
            true_ring = list(true_labels)[0] if true_labels else ""
            
            detected_rings.append({
                "ring_id": f"ring_found_{ring_counter:02d}",
                "member_accounts": members,
                "member_count": len(members),
                "total_rupees": round(total_rupees, 2),
                "time_first_flagged": earliest_flag,
                "matched_true_ring": true_ring
            })
            ring_counter += 1
            
    # Sort detected rings by total rupees (largest rings first)
    detected_rings.sort(key=lambda x: -x["total_rupees"])
    return detected_rings


def evaluate_ring_detection(detected_rings, all_decisions, round1_path):
    """
    Compares detected rings against ground-truth rings from round1.csv.
    Computes recall and average detection delay (seconds).
    """
    # 1. Identify all true rings present in the replayed test decisions
    true_rings_in_test = set()
    for tx in all_decisions:
        rid = tx.get("ring_id")
        if rid and rid != "nan" and rid != "":
            true_rings_in_test.add(rid)
            
    num_true_rings = len(true_rings_in_test)
    
    # 2. Extract true ring start times from round1.csv
    ring_start_times = {}
    if os.path.exists(round1_path):
        df_r1 = pd.read_csv(round1_path)
        for rid in true_rings_in_test:
            sub = df_r1[df_r1["ring_id"] == rid]
            if len(sub) > 0:
                ring_start_times[rid] = int(sub["time"].min())
                
    # 3. Match detected rings with ground truth
    found_true_rings = set()
    detection_delays = []
    
    for r in detected_rings:
        matched = r["matched_true_ring"]
        if matched and matched in true_rings_in_test:
            found_true_rings.add(matched)
            if matched in ring_start_times:
                delay = r["time_first_flagged"] - ring_start_times[matched]
                detection_delays.append(max(0, delay))
                
    avg_delay_sec = float(sum(detection_delays) / len(detection_delays)) if detection_delays else 0.0
    
    return {
        "num_true_rings": num_true_rings,
        "rings_found_count": len(found_true_rings),
        "true_rings_found": sorted(list(found_true_rings)),
        "avg_delay_sec": round(avg_delay_sec, 1)
    }


def main():
    print("=" * 70)
    print(" STEP 7: GRAPH-BASED MULE RING DETECTION")
    print("=" * 70)
    print(f"Loading decisions from {DECISIONS_PATH}...")
    flagged_txns, all_decisions, total_txns = load_flagged_decisions(DECISIONS_PATH)
    
    print(f"Total Transactions Scanned : {total_txns:,}")
    print(f"High-Risk Transactions     : {len(flagged_txns):,} (HOLD or STEP-UP)\n")
    
    print("Building account network graph & clustering connected components...")
    detected_rings = find_mule_rings(flagged_txns)
    
    # Save to data/rings.json
    os.makedirs(os.path.dirname(RINGS_OUTPUT), exist_ok=True)
    with open(RINGS_OUTPUT, "w") as f:
        json.dump(detected_rings, f, indent=2)
    print(f"Saved {len(detected_rings)} detected rings to {RINGS_OUTPUT}\n")
    
    # Evaluate against ground truth
    metrics = evaluate_ring_detection(detected_rings, all_decisions, ROUND1_PATH)
    
    print("=" * 70)
    print(" MULE RING DETECTION PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"True Planted Rings in Test Period : {metrics['num_true_rings']}")
    print(f"Planted Rings Successfully Found  : {metrics['rings_found_count']} of {metrics['num_true_rings']} ({(metrics['rings_found_count']/max(metrics['num_true_rings'],1))*100:.1f}%)")
    print(f"Detected Ring Clusters Total      : {len(detected_rings)}")
    print(f"Average Time to First Detection   : {metrics['avg_delay_sec']:.1f} seconds after ring start")
    print("-" * 70)
    
    if detected_rings:
        print("Top Detected Mule Rings:")
        for i, r in enumerate(detected_rings[:4]):
            true_tag = f"[Matches {r['matched_true_ring']}]" if r['matched_true_ring'] else ""
            print(f"  {r['ring_id']}: {r['member_count']} accounts | Rs {r['total_rupees']:,.2f} | Flagged at t={r['time_first_flagged']}s {true_tag}")
            print(f"    Sample Accounts: {', '.join(r['member_accounts'][:6])}...")
    print("=" * 70)
    print("\n[SUCCESS] Step 7 complete: Ring graph detection verified.")


if __name__ == "__main__":
    main()

