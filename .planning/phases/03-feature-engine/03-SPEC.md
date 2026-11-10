# Phase 03 Specification: Stateful Real-Time Feature Engine

## Goal
Build `src/features.py` to extract high-signal behavioral and temporal features from streaming UPI transactions using pure Python state structures (`dict`, `deque`). It maintains an $O(1)$ in-memory representation of account behavior without external graph/database bottlenecks.

## Core Behavioral Features per Payment
For every payment $P_t$, using **only** events strictly prior ($t' < t$):
1. **Account Ages**:
   - `sender_age_days`: Age of sender account since first seen.
   - `receiver_age_days`: Age of receiver account since first seen.
2. **Sender Behavior**:
   - `sender_amount_ratio`: Transaction amount divided by sender's historical median amount.
   - `is_new_receiver_for_sender`: Binary (1 if sender has never transferred to this receiver, 0 otherwise).
   - `sender_tx_count_last_hour`: Velocity count of payments sent by sender in the last 60 minutes.
   - `device_changed`: Binary (1 if payment device differs from primary registered device, 0 otherwise).
3. **Receiver Mule & Ring Signatures**:
   - `receiver_distinct_senders_24h`: Number of unique senders who paid this receiver in the last 24 hours.
   - `receiver_distinct_receivers_24h`: Number of unique receivers this account paid out to in the last 24 hours.
   - `receiver_pass_through_ratio_30m`: Money sent out in last 30 minutes divided by money received in last 30 minutes ($0.0$ if no inbound funds). A value close to $1.0$ indicates mule pass-through behavior.
   - `receiver_minutes_since_last_inflow`: Minutes elapsed since the previous incoming transfer to this receiver.

## Architecture & Operational Contract
- **Reusable Function**: Must expose `extract_and_update_state(tx, state)` returning the feature vector for a single payment. This exact function will be invoked during real-time scoring in `replay.py`.
- **Performance**: Pure Python dictionaries and `collections.deque`. No slow global graph computations or network queries. Target processing speed $> 20,000$ transactions/second.
- **Artifact Outputs**:
  - `data/features_round1.csv` (includes `is_fraud`)
  - `data/features_round2.csv` (includes `is_fraud`)
- **Verification Criteria**:
  - Displays runtime and processing throughput.
  - Prints 5 sample feature rows for a known mule ring collector, demonstrating high distinct senders and pass-through ratio $\approx 1.0$.

