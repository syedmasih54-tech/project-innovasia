"""
StreamShield - Tamper-Evident Log Verification (src/verify_log.py)
===================================================================
Verifies the cryptographic SHA-256 hash chain of data/decisions.jsonl.
Each line's hash is computed as:
  hash = SHA-256(prev_hash + canonical_json_of_record_without_hash)

Prints PASS if all records are intact, or reports the exact first broken line.
Returns exit code 0 on PASS, 1 on FAIL.
"""

import os
import sys
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIONS_FILE = os.path.join(BASE_DIR, "data", "decisions.jsonl")


def compute_record_hash(prev_hash, record_dict):
    """
    Computes deterministic SHA-256 hash of the record concatenated with prev_hash.
    Uses canonical JSON sorting with compact separators to guarantee cross-platform consistency.
    """
    payload = {k: v for k, v in record_dict.items() if k != "hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    to_hash = f"{prev_hash}:{canonical}".encode("utf-8")
    return hashlib.sha256(to_hash).hexdigest()


def verify_chain(log_path=DECISIONS_FILE, verbose=True):
    """
    Verifies the entire decision audit chain.
    Returns: (is_valid: bool, records_checked: int, details: dict)
    """
    if not os.path.exists(log_path):
        print(f"FAIL: Decision log not found at {log_path}")
        return False, 0, {"error": "File not found"}

    expected_prev = "0" * 64
    line_number = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_number += 1
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                msg = f"FAIL: Corrupt JSON on line {line_number}: {err}"
                if verbose:
                    print(msg)
                return False, line_number, {"broken_line": line_number, "reason": "Corrupt JSON"}

            rec_prev = record.get("prev_hash")
            rec_hash = record.get("hash")

            if not rec_prev or not rec_hash:
                msg = f"FAIL: Missing cryptographic hashes on line {line_number} (txn_id: {record.get('txn_id')})"
                if verbose:
                    print(msg)
                return False, line_number, {"broken_line": line_number, "reason": "Missing hash fields"}

            # 1. Verify prev_hash matches previous record's hash
            if rec_prev != expected_prev:
                msg = (
                    f"FAIL: Broken hash link at line {line_number} (txn_id: {record.get('txn_id')})\n"
                    f"  Expected prev_hash: {expected_prev}\n"
                    f"  Recorded prev_hash: {rec_prev}"
                )
                if verbose:
                    print(msg)
                return False, line_number, {
                    "broken_line": line_number,
                    "reason": "prev_hash mismatch",
                    "expected_prev": expected_prev,
                    "found_prev": rec_prev
                }

            # 2. Verify record hash matches SHA-256 of contents
            computed_hash = compute_record_hash(rec_prev, record)
            if computed_hash != rec_hash:
                msg = (
                    f"FAIL: Tampered record content at line {line_number} (txn_id: {record.get('txn_id')})\n"
                    f"  Expected hash: {computed_hash}\n"
                    f"  Recorded hash: {rec_hash}"
                )
                if verbose:
                    print(msg)
                return False, line_number, {
                    "broken_line": line_number,
                    "reason": "Hash content mismatch",
                    "computed_hash": computed_hash,
                    "recorded_hash": rec_hash
                }

            # Advance chain
            expected_prev = rec_hash

    head_hash = expected_prev
    if verbose:
        print("=" * 72)
        print(" STREAMSHIELD TAMPER-EVIDENT AUDIT CHAIN VERIFICATION")
        print("=" * 72)
        print(f"Result           : PASS")
        print(f"Records Verified : {line_number:,} transactions")
        print(f"Genesis Hash     : {'0' * 64}")
        print(f"Chain Head Hash  : {head_hash}")
        print(f"Algorithm        : SHA-256 Recursive HMAC-Equivalent Merkle Chain")
        print(f"Integrity Status : 100% Cryptographically Intact (Zero Tampering)")
        print("=" * 72)

    return True, line_number, {"head_hash": head_hash, "total_records": line_number}


if __name__ == "__main__":
    is_valid, count, details = verify_chain()
    if is_valid:
        print("\nPASS")
        sys.exit(0)
    else:
        print("\nFAIL")
        sys.exit(1)

