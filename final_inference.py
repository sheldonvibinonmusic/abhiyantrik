"""
ROGUE PROTOCOL - FINAL INFERENCE CLI
Authoritative standalone inference runner using verified model artefact.

Usage:
    python final_inference.py
    python final_inference.py --input test.csv --output submission_verified.csv
"""
import os
import sys
import re
import argparse
import joblib
import pandas as pd
import numpy as np
from check_submission import check

def parse_args():
    parser = argparse.ArgumentParser(description="Authoritative Rogue Protocol inference CLI.")
    parser.add_argument("--input", default="test.csv", help="Path to input CSV (e.g. test.csv)")
    parser.add_argument("--output", default="submission_verified.csv", help="Path to output predictions CSV")
    parser.add_argument("--model", default="model_artefact_verified.joblib", help="Path to serialized model artefact")
    return parser.parse_args()

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

def clean_text(m):
    m = str(m) if pd.notna(m) else ""
    if m.startswith('NINE:'):
        m = m[5:].strip()
    return m

def engineer_features(df, nominal_centers, nominal_scales, cat_cols):
    out = pd.DataFrame(index=df.index)
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
        print(f"Error: model artefact '{args.model}' not found.")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        sys.exit(1)
        
    print(f"Loading verified model artefact: {args.model}")
    bundle = joblib.load(args.model)
    
    selected_exp = bundle.get('selected_exp', 'EXP_B_STRUCTURED_HGB')
    threshold = bundle['selected_th']
    centers = bundle['nominal_centers']
    scales = bundle['nominal_scales']
    cat_cols = bundle['cat_cols']
    m1 = bundle['struct_m1']
    m2 = bundle['struct_m2']
    
    print(f"Reading input records from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input records loaded: {len(df)}")
    
    # 1. Structured predictions
    X_s = engineer_features(df, centers, scales, cat_cols)
    p1 = m1.predict_proba(X_s)[:, 1]
    p2 = m2.predict_proba(X_s)[:, 1]
    p_struct = 0.60 * p1 + 0.40 * p2
    
    # 2. Hybrid text predictions if applicable
    if selected_exp == 'EXP_C_HYBRID_BLEND' and bundle.get('text_pipeline') is not None:
        clean_msgs = df['message'].apply(clean_text)
        X_t = bundle['text_pipeline'].transform(clean_msgs)
        p_text = bundle['text_clf'].predict_proba(X_t)[:, 1]
        w_s = bundle.get('hybrid_weight_struct', 0.5)
        p_final = w_s * p_struct + (1.0 - w_s) * p_text
        print(f"Computed hybrid probabilities ({w_s:.2f} Structured + {1-w_s:.2f} Text).")
    else:
        p_final = p_struct
        print("Computed structured ensemble probabilities.")
        
    # Apply calibrated decision threshold
    verdicts = (p_final >= threshold).astype(int)
    
    # Construct output DataFrame
    sub = pd.DataFrame({
        'log_id': df['log_id'],
        'is_true_fault': verdicts
    })
    
    sub.to_csv(args.output, index=False)
    print(f"Exported {len(sub)} predictions to: {args.output}")
    print(f"Flagged true faults: {verdicts.sum()} ({verdicts.mean()*100:.2f}%)")
    print(f"Flagged noise/false alarms: {(verdicts == 0).sum()} ({(1-verdicts.mean())*100:.2f}%)")
    
    if args.input == "test.csv":
        print("\nRunning official submission checker:")
        check(args.output)

if __name__ == "__main__":
    main()
