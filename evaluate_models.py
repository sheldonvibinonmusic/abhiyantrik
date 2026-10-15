"""
ROGUE PROTOCOL - MODEL EVALUATION & COMPARISON RUNNER
Analyzes and outputs cross-model benchmark results across structured, hybrid, and transformer models.
"""
import os
import sys
import json
import pandas as pd
from check_submission import check

def main():
    print("=" * 75)
    print("ROGUE PROTOCOL: MODEL EVALUATION & ARCHITECTURAL BENCHMARK")
    print("=" * 75)
    
    report_path = 'artifacts/transformer_evaluation.json'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
    else:
        print("Transformer evaluation report not found yet.")
        
    print("\nChecking candidate submission files on disk:")
    for f in ['submission_verified.csv', 'submission_transformer_candidate.csv', 'submission.csv']:
        if os.path.exists(f):
            print(f"\n--- Checking {f} ---")
            check(f)
        else:
            print(f"File {f} does not exist.")

if __name__ == "__main__":
    main()
