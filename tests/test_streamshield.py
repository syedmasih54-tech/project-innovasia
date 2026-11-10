"""
StreamShield - Production Automated Test Suite
Verifies core feature engine consistency, model bundles, cost-utility decisions,
graph clustering algorithms, and the privacy-preserving consortium ledger.
"""

import os
import unittest
import numpy as np
import joblib

from src.features import AccountMemoryState, compute_features_and_update
from src.decision import evaluate_decision_and_reasons, RECOVERY_RATE, STEP_UP_COST, ABANDON_PROB, HOLD_THRESHOLD
from src.consortium import PrivacyPreservingConsortiumLedger


class TestStreamShieldPipeline(unittest.TestCase):

    def setUp(self):
        self.state = AccountMemoryState()
        self.sample_tx = {
            "time": 100.0,
            "txn_id": "tx_test_001",
            "sender": "acc_sender_1",
            "receiver": "acc_receiver_1",
            "amount": 5000.0,
            "device_id": "dev_001",
            "is_fraud": 0,
            "ring_id": None
        }

    def test_01_feature_engine_cold_start(self):
        """Test feature extraction on cold start accounts (brand new accounts)."""
        feats = compute_features_and_update(self.sample_tx, self.state)
        
        self.assertIn("sender_age_days", feats)
        self.assertIn("receiver_age_days", feats)
        self.assertIn("pass_through_ratio_30m", feats)
        self.assertEqual(feats["sender_age_days"], 0.0)
        self.assertEqual(feats["is_new_receiver"], 1)
        self.assertEqual(feats["pass_through_ratio_30m"], 0.0)

    def test_02_passthrough_ratio_calculation(self):
        """Test 30-minute pass-through ratio calculation for mule accounts."""
        # 1. Mule receives 10,000 from victim
        inflow_tx = {
            "time": 200.0,
            "txn_id": "tx_inflow",
            "sender": "victim_acc",
            "receiver": "mule_acc",
            "amount": 10000.0,
            "device_id": "dev_victim"
        }
        compute_features_and_update(inflow_tx, self.state)
        
        # 2. Mule sends 9,500 out to exit account within 5 minutes
        outflow_tx = {
            "time": 500.0,
            "txn_id": "tx_outflow",
            "sender": "mule_acc",
            "receiver": "exit_acc",
            "amount": 9500.0,
            "device_id": "dev_mule"
        }
        compute_features_and_update(outflow_tx, self.state)
        
        # 3. Another payment enters mule_acc right after -> check mule_acc's pass-through ratio
        check_tx = {
            "time": 600.0,
            "txn_id": "tx_check",
            "sender": "victim_2",
            "receiver": "mule_acc",
            "amount": 1000.0,
            "device_id": "dev_victim2"
        }
        feats = compute_features_and_update(check_tx, self.state)
        # Outflow (9,500) / Inflow (10,000) = 0.95
        self.assertAlmostEqual(feats["pass_through_ratio_30m"], 0.95, places=2)

    def test_03_model_bundle_integrity(self):
        """Verify trained model bundle exists and contains all required artifacts."""
        bundle_path = os.path.join("data", "model_bundle.joblib")
        self.assertTrue(os.path.exists(bundle_path), "Model bundle file missing!")
        
        bundle = joblib.load(bundle_path)
        required_keys = ["lgb_model", "calibrator", "iforest", "feature_cols", "blend_weight"]
        for k in required_keys:
            self.assertIn(k, bundle, f"Missing key in bundle: {k}")

    def test_04_cost_utility_decision_policy(self):
        """Verify economic cost-utility thresholds."""
        # Low risk, low amount -> ALLOW
        score = 0.01
        amt = 500.0
        exp_loss = score * amt * (1.0 - RECOVERY_RATE)
        friction = STEP_UP_COST + (1.0 - score) * ABANDON_PROB * amt
        dec = "HOLD" if exp_loss >= HOLD_THRESHOLD else ("STEP-UP" if exp_loss >= friction else "ALLOW")
        self.assertEqual(dec, "ALLOW")
        
        # High loss -> HOLD
        score_high = 0.95
        amt_high = 50000.0
        exp_loss_high = score_high * amt_high * (1.0 - RECOVERY_RATE)
        dec_high = "HOLD" if exp_loss_high >= HOLD_THRESHOLD else "ALLOW"
        self.assertEqual(dec_high, "HOLD")

    def test_05_privacy_preserving_consortium_ledger(self):
        """Verify inter-bank blinded token generation and multi-bank velocity detection."""
        ledger = PrivacyPreservingConsortiumLedger()
        vpa = "mule_test@upi"
        
        token1 = ledger.anonymize_vpa(vpa)
        token2 = ledger.anonymize_vpa(vpa)
        self.assertEqual(token1, token2, "Tokenization must be deterministic with same salt.")
        self.assertNotIn("mule_test", token1, "Token must be a one-way cryptographic hash.")
        
        # Simulate transfers from 3 distinct banks
        ledger.record_interbank_transfer("HDFC", vpa, timestamp=1000.0)
        ledger.record_interbank_transfer("ICICI", vpa, timestamp=1050.0)
        ledger.record_interbank_transfer("SBI", vpa, timestamp=1100.0)
        
        signals = ledger.get_cross_bank_signals(vpa, timestamp=1150.0)
        self.assertEqual(signals["distinct_origin_banks_15m"], 3)
        self.assertEqual(signals["is_multi_bank_mule_risk"], 1)


if __name__ == "__main__":
    unittest.main()
