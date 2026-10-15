"""
ROGUE PROTOCOL - MANDATORY SUBMISSION VERIFICATION SCRIPT
Runs exhaustive structural, ID integrity, prediction integrity, and official checker validations.
"""
import sys
import os
import pandas as pd
import numpy as np
from check_submission import check

def verify_file(sub_path="submission_verified.csv", test_path="test.csv"):
    print("=" * 75)
    print(f"RUNNING MANDATORY SUBMISSION VERIFICATION ON: {sub_path}")
    print("=" * 75)
    
    if not os.path.exists(sub_path):
        print(f"[FAIL] Submission file '{sub_path}' does not exist on disk.")
        return False
        
    if not os.path.exists(test_path):
        print(f"[FAIL] Test reference file '{test_path}' does not exist on disk.")
        return False
        
    # 1. Read files
    try:
        sub = pd.read_csv(sub_path)
        test = pd.read_csv(test_path)
    except Exception as e:
        print(f"[FAIL] Error reading CSV: {e}")
        return False
        
    all_passed = True
    
    # 2. Check column names and ordering
    expected_cols = ["log_id", "is_true_fault"]
    if list(sub.columns) == expected_cols:
        print(f"[PASS] Columns exact match: {list(sub.columns)}")
    else:
        print(f"[FAIL] Columns mismatch. Expected {expected_cols}, got {list(sub.columns)}")
        all_passed = False
        
    # 3. Check row count
    if len(sub) == 3000:
        print(f"[PASS] Exact row count verified: 3,000 data rows")
    else:
        print(f"[FAIL] Row count mismatch. Expected 3000, got {len(sub)}")
        all_passed = False
        
    # 4. Check ID alignment and integrity
    sub_ids = list(sub['log_id'])
    test_ids = list(test['log_id'])
    
    if sub_ids == test_ids:
        print("[PASS] ID integrity: 1-to-1 exact sequential match with test.csv")
    else:
        print("[FAIL] ID mismatch with test.csv")
        missing = set(test_ids) - set(sub_ids)
        extra = set(sub_ids) - set(test_ids)
        if missing:
            print(f"  Missing IDs: {len(missing)}")
        if extra:
            print(f"  Extra IDs: {len(extra)}")
        all_passed = False
        
    if sub['log_id'].duplicated().any():
        print(f"[FAIL] Found {sub['log_id'].duplicated().sum()} duplicate IDs!")
        all_passed = False
    else:
        print("[PASS] Zero duplicate log_ids found.")
        
    # 5. Check prediction values
    null_count = sub['is_true_fault'].isna().sum()
    if null_count == 0:
        print("[PASS] Zero null or NaN predictions.")
    else:
        print(f"[FAIL] Found {null_count} null predictions!")
        all_passed = False
        
    is_binary = sub['is_true_fault'].isin([0, 1]).all()
    if is_binary:
        print("[PASS] Strict binary encoding verified (only integers 0 and 1).")
    else:
        print("[FAIL] Predictions contain values other than 0 and 1!")
        all_passed = False
        
    # 6. Class distribution summary
    n_ones = int((sub['is_true_fault'] == 1).sum())
    n_zeros = int((sub['is_true_fault'] == 0).sum())
    total = n_ones + n_zeros
    
    print(f"\nPrediction Distribution Summary:")
    print(f"  Total Flagged Faults (1):      {n_ones:,} ({n_ones/total*100:.2f}%)")
    print(f"  Total Non-Faults / Noise (0): {n_zeros:,} ({n_zeros/total*100:.2f}%)")
    print(f"  Total Verified Predictions:   {total:,}")
    
    if total != 3000:
        print(f"[FAIL] Total counts ({total}) do not sum to 3,000!")
        all_passed = False
    else:
        print("[PASS] Total predictions sum exactly to 3,000.")
        
    # 7. Official format checker
    print("\nRunning Official Competition Checker:")
    official_check = check(sub_path)
    if official_check:
        print("[PASS] Official check_submission.py passed successfully!")
    else:
        print("[FAIL] Official check_submission.py failed.")
        all_passed = False
        
    print("\n" + "=" * 75)
    if all_passed:
        print("FINAL VERDICT: ALL MANDATORY VALIDATION CHECKS PASSED [OK]")
    else:
        print("FINAL VERDICT: SUBMISSION FAILED VALIDATION CHECKS [X]")
    print("=" * 75)
    return all_passed

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "submission_verified.csv"
    success = verify_file(target)
    sys.exit(0 if success else 1)
