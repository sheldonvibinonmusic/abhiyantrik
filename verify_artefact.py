import joblib
import pandas as pd
import numpy as np
import re
from check_submission import check

print("Testing Model Artefact Loading and Inference...")
artefacts = joblib.load('model_artefact.joblib')

hgb_reg = artefacts['hgb_reg']
hgb_weighted = artefacts['hgb_weighted']
threshold = artefacts['threshold']
NOMINAL_CENTERS = artefacts['nominal_centers']
NOMINAL_SCALES = artefacts['nominal_scales']
hgb_cats = artefacts['cat_cols']

print(f"Loaded models successfully. Threshold = {threshold}")

# Test on test.csv
test = pd.read_csv('test.csv')

def parse_and_engineer(df):
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
        if pd.isna(r) or mod not in NOMINAL_CENTERS:
            z_devs.append(0.0)
            abs_z_devs.append(0.0)
            sq_z_devs.append(0.0)
            is_extreme.append(0)
        else:
            z = (r - NOMINAL_CENTERS[mod]) / NOMINAL_SCALES[mod]
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
    
    for c in hgb_cats:
        out[c] = out[c].astype('category')
    return out

X_test = parse_and_engineer(test)
p1 = hgb_reg.predict_proba(X_test)[:, 1]
p2 = hgb_weighted.predict_proba(X_test)[:, 1]
p_blend = 0.60 * p1 + 0.40 * p2

preds = (p_blend >= threshold).astype(int)

# Check against submission.csv on disk
sub_disk = pd.read_csv('submission.csv')
match = (sub_disk['is_true_fault'] == preds).all()
print(f"Predictions exactly match submission.csv on disk: {match}")
print(f"Total verdicts: {len(preds)}, Flagged: {preds.sum()}")

check_result = check("submission.csv")
print(f"Format check result: {check_result}")
