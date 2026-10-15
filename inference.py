"""
ROGUE PROTOCOL - ROUND 1
Inference Script for Recalibration Unit

Usage:
    python inference.py
    python inference.py --input test.csv --output submission.csv
"""
import argparse
import sys
import os
import re
import joblib
import pandas as pd
import numpy as np
from check_submission import check

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Rogue Protocol predictions from model artefact.")
    parser.add_argument("--input", default="test.csv", help="Path to input test CSV")
    parser.add_argument("--output", default="submission.csv", help="Path to output submission CSV")
    parser.add_argument("--model", default="model_artefact.joblib", help="Path to saved model artefact")
    return parser.parse_args()

def extract_features(df, nominal_centers, nominal_scales, cat_cols):
    out = pd.DataFrame(index=df.index)
    
    def parse_msg(m):
        m = str(m) if pd.notna(m) else ""
        bracket = re.search(r'\[(.*?)\]', m)
        bracket_content = bracket.group(1) if bracket else ""
        text_before = m[:bracket.start()].strip() if bracket else m.strip()
        if text_before.startswith('NINE:'):
            text_before = text_before[5:].strip()
        parts = text_before.split('. ', 1)
        if len(parts) == 2:
            prefix, payload = parts[0].strip(), parts[1].strip()
        else:
            prefix, payload = "none", parts[0].strip()
        if not prefix:
            prefix = "none"
        return prefix, payload, bracket_content

    prefixes, payloads, brackets = zip(*df['message'].apply(parse_msg))
    out['prefix'] = list(prefixes)
    out['payload'] = list(payloads)
    out['is_redacted'] = [1 if 'REDACTED' in b else 0 for b in brackets]
    out['reading_isna'] = df['reading'].isna().astype(int)
    out['reading'] = df['reading']
    
    z_devs = []
    abs_z_devs = []
    sq_z_devs = []
    is_extreme = []
    for mod, r in zip(df['module'], df['reading']):
        if pd.isna(r) or mod not in nominal_centers:
            z_devs.append(0.0)
            abs_z_devs.append(0.0)
            sq_z_devs.append(0.0)
            is_extreme.append(0)
        else:
            z = (r - nominal_centers[mod]) / nominal_scales[mod]
            z_devs.append(z)
            abs_z_devs.append(abs(z))
            sq_z_devs.append(z ** 2)
            is_extreme.append(1 if abs(z) > 1.5 else 0)
            
    out['z_dev'] = z_devs
    out['abs_z_dev'] = abs_z_devs
    out['sq_z_dev'] = sq_z_devs
    out['is_extreme_reading'] = is_extreme
    out['delay_ms'] = df['delay_ms']
    out['delay_log'] = np.log1p(df['delay_ms'].clip(lower=0))
    out['delay_high'] = (df['delay_ms'] > 450).astype(int)
    out['retry_count'] = df['retry_count']
    out['has_retries'] = (df['retry_count'] > 0).astype(int)
    out['module'] = df['module']
    out['mode'] = df['mode']
    out['node'] = df['node']
    out['status'] = df['status']
    
    calm = out['prefix'].isin(['all is well', 'crew need not worry', 'nothing to see'])
    severe = out['status'].isin(['CRIT', 'ERROR'])
    out['calm_yet_severe'] = (calm & severe).astype(int)
    
    benign_payloads = [
        'scheduled calibration', 'self-test pattern', 'duplicate of earlier alert',
        'simulated drill event', 'recovered after retry', 'sensor warming up'
    ]
    out['is_benign_routine'] = out['payload'].isin(benign_payloads).astype(int)
    out['benign_yet_severe'] = (out['is_benign_routine'] & severe).astype(int)
    
    failure_payloads = [
        'dish actuator stalled', 'carrier drift beyond tolerance', 'phase lock lost',
        'breaker tripped twice', 'overvolt on rail B', 'cascade brownout detected',
        'O2 partial pressure falling', 'seal breach suspected', 'scrubber saturation',
        'bus arbitration deadlock', 'queue overflow, packets dropped', 'heartbeat missing from 3 nodes'
    ]
    out['is_failure_payload'] = out['payload'].isin(failure_payloads).astype(int)
    
    for c in cat_cols:
        out[c] = out[c].astype('category')
    return out

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"Error: model file {args.model} not found.")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} not found.")
        sys.exit(1)
        
    print(f"Loading model artefact from {args.model}...")
    artefacts = joblib.load(args.model)
    hgb_reg = artefacts['hgb_reg']
    hgb_weighted = artefacts['hgb_weighted']
    threshold = artefacts['threshold']
    nominal_centers = artefacts['nominal_centers']
    nominal_scales = artefacts['nominal_scales']
    cat_cols = artefacts['cat_cols']
    
    print(f"Reading input data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Extracted features for {len(df)} rows...")
    X = extract_features(df, nominal_centers, nominal_scales, cat_cols)
    
    print("Computing ensemble probabilities...")
    p1 = hgb_reg.predict_proba(X)[:, 1]
    p2 = hgb_weighted.predict_proba(X)[:, 1]
    p_blend = 0.60 * p1 + 0.40 * p2
    
    verdicts = (p_blend >= threshold).astype(int)
    sub = pd.DataFrame({
        'log_id': df['log_id'],
        'is_true_fault': verdicts
    })
    
    print(f"Writing verdicts to {args.output}...")
    sub.to_csv(args.output, index=False)
    print(f"Exported {len(sub)} rows. Flagged faults: {verdicts.sum()} ({verdicts.mean()*100:.1f}%)")
    
    if args.input == "test.csv":
        print("Running official format check:")
        check(args.output)

if __name__ == "__main__":
    main()
