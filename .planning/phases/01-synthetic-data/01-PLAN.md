# Phase 01 Plan: Synthetic Data Generation Engine

## Goal
Implement `src/generator.py` to generate realistic synthetic UPI payment transaction logs simulating legitimate consumer behavior, account cold starts, coordinated mule rings, and scam payments across two rounds:
- `data/round1.csv`: Standard mule rings (rapid fan-in to collector + swift fan-out) and high-value scam payments.
- `data/round2.csv`: Adversarial concept drift with delayed forwarding, split payments (smurfing), layered chains ($A \to B \to C \to D$), and rotated device IDs.

## Requirements Checklist
1. **Configurable Hyperparameters**: Place all dataset sizing numbers at the top of `src/generator.py` (e.g. `NUM_ACCOUNTS = 20000`, `NUM_PAYMENTS = 400000`, `NUM_RINGS = 20`, `NUM_SCAMS = 150`, `RANDOM_SEED = 42`).
2. **Round 1 Generation**:
   - 20,000 accounts with ~15% cold-start (first transaction appearing in the second half of the 14-day window).
   - ~400,000 normal transactions over 14 days ($14 \times 86,400\text{ seconds}$).
   - Skewed payment amounts (log-normal distribution: majority small UPI transfers, few high-value).
   - Realistic recipient affinity (senders route majority of payments to 2–5 preferred receivers).
   - 20 Mule Rings: 6–12 accounts funnel funds to 1 collector within 10 minutes; collector passes $\ge 90\%$ onward to 1–2 exit accounts within 5 minutes.
   - 150 Scam Payments: Uncharacteristically large payments to previously unseen receivers.
   - Fraud rate strictly $< 0.1\%$ of total transactions.
3. **Round 2 Generation (Adversarial Drift)**:
   - Same account universe and baseline volume.
   - Delayed forwarding: Collector waits 30 to 120 minutes.
   - Smurfing / structuring: Outbound funds split into multiple smaller tranches.
   - Multi-hop chains: Account $A \to B \to C \to D$.
   - Device rotation: Mule accounts execute from newly registered device IDs.
4. **Data Schema**:
   - Columns: `txn_id`, `time`, `sender`, `receiver`, `amount`, `device_id`, `channel`, `is_fraud`, `ring_id`.
   - Chronologically sorted by `time`.
5. **CLI Verification**:
   - Executes with `python src/generator.py`.
   - Prints: Total rows, fraud percentage, number of rings, and 5 example rows of one mule ring.

