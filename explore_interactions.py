import pandas as pd
import numpy as np
import re
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

# Feature Engineering function
def extract_features(df):
    f = pd.DataFrame(index=df.index)
    
    # 1. Nominal centers and normalized deviations per module
    nominal_centers = {
        'Lab-Power': 12.18,
        'Life-Support': 21.11,
        'Antenna': 43.13,
        'Notice-Bus': 313.09
    }
    nominal_scales = {
        'Lab-Power': 1.05,
        'Life-Support': 1.67,
        'Antenna': 7.95,
        'Notice-Bus': 56.31
    }
    
    f['reading_raw'] = df['reading']
    f['reading_isna'] = df['reading'].isna().astype(int)
    
    # Deviation from nominal center
    devs = []
    abs_devs = []
    z_devs = []
    for mod, r in zip(df['module'], df['reading']):
        if pd.isna(r) or mod not in nominal_centers:
            devs.append(np.nan)
            abs_devs.append(np.nan)
            z_devs.append(np.nan)
        else:
            dev = r - nominal_centers[mod]
            scale = nominal_scales[mod]
            devs.append(dev)
            abs_devs.append(abs(dev))
            z_devs.append(dev / scale)
            
    f['dev_from_nominal'] = devs
    f['abs_dev_from_nominal'] = abs_devs
    f['z_dev_from_nominal'] = z_devs
    
    # Extreme deviation flags per module
    is_extreme = []
    for mod, r in zip(df['module'], df['reading']):
        if pd.isna(r):
            is_extreme.append(0)
        elif mod == 'Lab-Power':
            is_extreme.append(int(r < 10.74 or r > 13.34))
        elif mod == 'Life-Support':
            is_extreme.append(int(r < 18.96 or r > 22.90))
        elif mod == 'Antenna':
            is_extreme.append(int(r < 31.92 or r > 51.27))
        elif mod == 'Notice-Bus':
            is_extreme.append(int(r < 239.72 or r > 374.64))
        else:
            is_extreme.append(0)
    f['is_extreme_reading'] = is_extreme
    
    # 2. Categoricals encoded
    for col in ['module', 'mode', 'node', 'status']:
        f[col] = df[col].astype('category')
        
    # 3. Delays & retries
    f['delay_ms'] = df['delay_ms']
    f['delay_high'] = (df['delay_ms'] > 450).astype(int)
    f['retry_count'] = df['retry_count']
    f['has_retries'] = (df['retry_count'] > 0).astype(int)
    
    # 4. Message features
    msg = df['message'].fillna('')
    f['msg_len'] = msg.apply(len)
    
    # Calm/Reassurance phrases (NINE lying / burying faults)
    f['calm_phrase'] = msg.apply(lambda x: int(any(p in x for p in [
        'all is well', 'nothing to see', 'crew need not worry', 'no cause for concern'
    ])))
    
    # False alarm / benign phrases (NINE crying wolf or benign routine)
    f['benign_routine'] = msg.apply(lambda x: int(any(p in x for p in [
        'simulated drill', 'sensor warming up', 'duplicate of earlier',
        'scheduled calibration', 'self-test pattern', 'recovered after retry'
    ])))
    
    # Real failure indicators in message
    f['failure_keywords'] = msg.apply(lambda x: int(any(p in x for p in [
        'breaker tripped', 'phase lock lost', 'packets dropped', 'overflow',
        'falling', 'breach', 'stalled', 'deadlock', 'overvolt', 'brownout', 'missing from'
    ])))
    
    # Trailing unit extracted from bracket
    def get_unit(m):
        match = re.search(r'\[.*?\s+(dBm|%O2|msg/s|V)\]', m)
        return match.group(1) if match else 'None'
    f['unit'] = msg.apply(get_unit).astype('category')
    
    return f

X_train = extract_features(train)
y_train = train['is_true_fault']
X_val = extract_features(val)
y_val = val['is_true_fault']

print("Features extracted:")
print(X_train.head())

# Train a HistGradientBoostingClassifier (handles missing values and categoricals natively)
cat_cols = ['module', 'mode', 'node', 'status', 'unit']
hgb = HistGradientBoostingClassifier(
    categorical_features=cat_cols,
    random_state=42,
    max_iter=150,
    learning_rate=0.08
)

hgb.fit(X_train, y_train)
val_prob = hgb.predict_proba(X_val)[:, 1]

# Tune threshold on val
best_th = 0.5
best_f1 = 0
for th in np.arange(0.2, 0.8, 0.02):
    pred = (val_prob >= th).astype(int)
    score = f1_score(y_val, pred)
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"\nHistGradientBoosting Validation:")
print(f"Default th=0.5 F1: {f1_score(y_val, (val_prob >= 0.5).astype(int)):.4f}")
print(f"Best th={best_th:.2f} F1: {best_f1:.4f}")
pred_best = (val_prob >= best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred_best).ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Precision: {precision_score(y_val, pred_best):.4f}, Recall: {recall_score(y_val, pred_best):.4f}")
