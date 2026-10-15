import os
import sys
import hashlib
import json
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from check_submission import check

print("=" * 70)
print("PHASE 1: REPRODUCE AND AUDIT")
print("=" * 70)

# 1. Environment Details
print("1. PYTHON ENVIRONMENT")
print("Python executable:", sys.executable)
print("Python version:   ", sys.version.split()[0])
import sklearn, scipy
print("sklearn version:  ", sklearn.__version__)
print("pandas version:   ", pd.__version__)
print("numpy version:    ", np.__version__)
print("scipy version:    ", scipy.__version__)

# 2. Dataset Paths and Hashes
print("\n2. DATASET PATHS AND HASHES")
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = [
    'train.csv', 'val.csv', 'test.csv', 
    'submission.csv', 'submission_exp3_final.csv', 'submission_starter.csv',
    'model_artefact.joblib'
]
for f in files_to_hash:
    if os.path.exists(f):
        print(f"{f:<25} size: {os.path.getsize(f):>8} bytes | SHA256: {file_hash(f)}")

# 3. CSV Structural Inspection (First 5 Rows Only)
print("\n3. CSV STRUCTURAL INSPECTION (FIRST 5 ROWS)")
for fname in ['train.csv', 'val.csv', 'test.csv', 'submission.csv']:
    df_sample = pd.read_csv(fname, nrows=5)
    print(f"\n--- {fname} (Sample: 5 rows, {len(pd.read_csv(fname, usecols=[0]))} total rows, {len(df_sample.columns)} cols) ---")
    print("Columns & dtypes:")
    for col, dt in df_sample.dtypes.items():
        print(f"  {col:<15} {str(dt):<10} sample: {df_sample[col].iloc[0]}")

# 4. Concise Aggregate Statistics
print("\n4. CONCISE AGGREGATE STATISTICS")
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

print(f"Train rows: {len(train):,}, Fault rate: {train['is_true_fault'].mean():.4f} ({train['is_true_fault'].sum()} positive)")
print(f"Val rows:   {len(val):,}, Fault rate: {val['is_true_fault'].mean():.4f} ({val['is_true_fault'].sum()} positive)")
print(f"Test rows:  {len(test):,}")

print("\nMissing values:")
print(f"Train: {dict(train.isnull().sum()[train.isnull().sum() > 0])}")
print(f"Val:   {dict(val.isnull().sum()[val.isnull().sum() > 0])}")
print(f"Test:  {dict(test.isnull().sum()[test.isnull().sum() > 0])}")

print("\nCategorical distributions in train:")
for c in ['module', 'status', 'mode']:
    print(f"  {c}: {dict(train[c].value_counts())}")

# 5. Reproducing Baseline (Protocol v0)
print("\n5. REPRODUCING BASELINE (PROTOCOL v0)")
starter_features = ['reading', 'delay_ms', 'retry_count']
starter_pipe = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), starter_features)
    ),
    LogisticRegression(random_state=42)
)
starter_pipe.fit(train, train['is_true_fault'])
val_prob_base = starter_pipe.predict_proba(val)[:, 1]
val_pred_base = (val_prob_base >= 0.50).astype(int)

f1_base = f1_score(val['is_true_fault'], val_pred_base)
prec_base = precision_score(val['is_true_fault'], val_pred_base, zero_division=0)
rec_base = recall_score(val['is_true_fault'], val_pred_base, zero_division=0)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(val['is_true_fault'], val_pred_base).ravel()

print(f"Starter Model on val.csv (th=0.50):")
print(f"  F1: {f1_base:.4f}, Precision: {prec_base:.4f}, Recall: {rec_base:.4f}")
print(f"  TP: {tp_b}, FP: {fp_b}, FN: {fn_b}, TN: {tn_b}")
print(f"  Positive predictions on val: {val_pred_base.sum()}")

# 6. Reproducing Current Best Model from model_artefact.joblib
print("\n6. REPRODUCING CURRENT BEST MODEL FROM model_artefact.joblib")
artefacts = joblib.load('model_artefact.joblib')
hgb_reg = artefacts['hgb_reg']
hgb_weighted = artefacts['hgb_weighted']
saved_th = artefacts['threshold']
NOMINAL_CENTERS = artefacts['nominal_centers']
NOMINAL_SCALES = artefacts['nominal_scales']
cat_cols = artefacts['cat_cols']

def extract_features(df):
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
    
    for c in cat_cols:
        out[c] = out[c].astype('category')
    return out

X_val = extract_features(val)
p1_val = hgb_reg.predict_proba(X_val)[:, 1]
p2_val = hgb_weighted.predict_proba(X_val)[:, 1]
p_blend_val = 0.60 * p1_val + 0.40 * p2_val

for th in [0.50, 0.45, saved_th]:
    pred = (p_blend_val >= th).astype(int)
    f1 = f1_score(val['is_true_fault'], pred)
    prec = precision_score(val['is_true_fault'], pred)
    rec = recall_score(val['is_true_fault'], pred)
    tn, fp, fn, tp = confusion_matrix(val['is_true_fault'], pred).ravel()
    desc = "saved threshold" if th == saved_th else ("OOF-optimal" if th == 0.45 else "default")
    print(f"\nCurrent Best Model on val.csv at th={th:.4f} ({desc}):")
    print(f"  F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  Positive predictions on val: {pred.sum()} / 1000 ({pred.mean()*100:.1f}%)")

# 7. Verifying Current submission.csv
print("\n7. VERIFYING CURRENT submission.csv")
sub = pd.read_csv('submission.csv')
print(f"Row count: {len(sub)}")
print(f"Columns:   {list(sub.columns)}")
print(f"Flagged true faults in test: {sub['is_true_fault'].sum()} ({sub['is_true_fault'].mean()*100:.1f}%)")
print(f"Check submission result:")
check_valid = check('submission.csv')
print(f"Format checker valid: {check_valid}")

# Check test predictions alignment with saved model
X_test = extract_features(test)
p1_test = hgb_reg.predict_proba(X_test)[:, 1]
p2_test = hgb_weighted.predict_proba(X_test)[:, 1]
p_blend_test = 0.60 * p1_test + 0.40 * p2_test
test_pred_from_model = (p_blend_test >= saved_th).astype(int)
match_disk = (sub['is_true_fault'] == test_pred_from_model).all()
print(f"submission.csv strictly matches model_artefact predictions: {match_disk}")

# 8. Feature Path Inconsistency Audit
print("\n8. FEATURE PATH AUDIT ACROSS SCRIPTS")
print("Comparing feature definitions in run_experiments.py, test_pipeline.py, inference.py, starter_round1.ipynb:")
scripts_to_check = ['run_experiments.py', 'test_pipeline.py', 'inference.py', 'verify_artefact.py']
for s in scripts_to_check:
    with open(s, 'r') as f_in:
        content = f_in.read()
    has_nominal = 'NOMINAL_CENTERS' in content or 'nominal_centers' in content
    has_calm = 'calm_yet_severe' in content
    has_routine = 'is_benign_routine' in content
    has_fail = 'is_failure_payload' in content
    has_zdev = 'z_dev' in content
    print(f"  {s:<22}: nominal={has_nominal}, calm={has_calm}, routine={has_routine}, fail_payload={has_fail}, z_dev={has_zdev}")
