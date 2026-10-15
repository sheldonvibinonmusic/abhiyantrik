import pandas as pd
import numpy as np
import re
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

print("=" * 70)
print("PHASE 2: DATA AND LABEL DISCOVERY")
print("=" * 70)

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

# --- 1. Syntax Decomposition on Train ---
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

prefixes, payloads, brackets = zip(*train['message'].apply(parse_msg))
train['prefix'] = prefixes
train['payload'] = payloads
train['is_redacted'] = [1 if 'REDACTED' in b else 0 for b in brackets]

# --- 2. Label Rates by Categorical Features ---
print("\n--- 2. CATEGORICAL FEATURE LABEL RATES (TRAIN ONLY) ---")
for col in ['module', 'status', 'mode', 'node', 'prefix']:
    print(f"\nLabel rates by {col}:")
    agg = train.groupby(col)['is_true_fault'].agg(['count', 'sum', 'mean'])
    agg['negatives'] = agg['count'] - agg['sum']
    agg = agg.rename(columns={'sum': 'positives', 'mean': 'fault_rate'})
    print(agg[['count', 'positives', 'negatives', 'fault_rate']].sort_values('count', ascending=False))

# --- 3. Payload Distribution & Support ---
print("\n--- 3. TOP PAYLOAD TEMPLATES (TRAIN ONLY) ---")
p_agg = train.groupby('payload')['is_true_fault'].agg(['count', 'sum', 'mean'])
p_agg['negatives'] = p_agg['count'] - p_agg['sum']
p_agg = p_agg.rename(columns={'sum': 'positives', 'mean': 'fault_rate'})
print("Highest fault rate payloads:")
print(p_agg.sort_values('fault_rate', ascending=False).head(12))
print("\nLowest fault rate payloads (false alarms / noise):")
print(p_agg.sort_values('fault_rate', ascending=True).head(12))

# --- 4. Asymmetric Physical Sensor Boundaries per Module ---
print("\n--- 4. PER-MODULE SENSOR ANOMALY INFLECTION POINTS ---")
sensor_rules = {}
for mod in train['module'].unique():
    sub = train[train['module'] == mod]
    # Filter out missing/redacted
    sub_r = sub.dropna(subset=['reading'])
    f0 = sub_r[sub_r['is_true_fault'] == 0]['reading']
    f1 = sub_r[sub_r['is_true_fault'] == 1]['reading']
    
    # 5th and 95th percentiles of normal (non-fault) operations
    q05 = f0.quantile(0.05)
    q95 = f0.quantile(0.95)
    q01 = f0.quantile(0.01)
    q99 = f0.quantile(0.99)
    median_norm = f0.median()
    std_norm = f0.std()
    
    # Fault rates outside normal bounds
    low_rate = sub_r[sub_r['reading'] < q05]['is_true_fault'].mean()
    mid_rate = sub_r[(sub_r['reading'] >= q05) & (sub_r['reading'] <= q95)]['is_true_fault'].mean()
    high_rate = sub_r[sub_r['reading'] > q95]['is_true_fault'].mean()
    
    print(f"\nModule: {mod}")
    print(f"  Non-fault median={median_norm:.2f}, std={std_norm:.2f}")
    print(f"  Bounds: [q05={q05:.2f}, q95={q95:.2f}] (extreme: [q01={q01:.2f}, q99={q99:.2f}])")
    print(f"  Fault rate below q05: {low_rate:.4f} (count={len(sub_r[sub_r['reading'] < q05])})")
    print(f"  Fault rate nominal [q05, q95]: {mid_rate:.4f} (count={len(sub_r[(sub_r['reading'] >= q05) & (sub_r['reading'] <= q95)])})")
    print(f"  Fault rate above q95: {high_rate:.4f} (count={len(sub_r[sub_r['reading'] > q95])})")
    sensor_rules[mod] = {'median': median_norm, 'std': std_norm, 'low_b': q05, 'high_b': q95}

# --- 5. Candidate Rule Out-of-Fold Evaluation ---
print("\n--- 5. EVALUATING CANDIDATE RULES ON 5-FOLD CV ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Candidate Rule A: Redacted Sensor Reading
print("\nRule A: is_redacted == 1 -> is_true_fault = 1")
precs, recs, f1s = [], [], []
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    fold_val = train.iloc[val_i]
    pred = fold_val['is_redacted']
    precs.append(precision_score(fold_val['is_true_fault'], pred, zero_division=0))
    recs.append(recall_score(fold_val['is_true_fault'], pred, zero_division=0))
    f1s.append(f1_score(fold_val['is_true_fault'], pred, zero_division=0))
print(f"  Support: {train['is_redacted'].sum()} rows (Pos: {train[train['is_redacted']==1]['is_true_fault'].sum()})")
print(f"  CV Precision: {np.mean(precs):.4f}, Recall: {np.mean(recs):.4f}, F1: {np.mean(f1s):.4f}")

# Candidate Rule B: Calm prefix AND (CRIT or ERROR)
calm_prefixes = ['all is well', 'crew need not worry', 'nothing to see']
severe_status = ['CRIT', 'ERROR']
train['calm_severe'] = (train['prefix'].isin(calm_prefixes) & train['status'].isin(severe_status)).astype(int)
print("\nRule B: calm_prefix & severe_status -> is_true_fault = 1")
precs, recs, f1s = [], [], []
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    fold_val = train.iloc[val_i]
    pred = fold_val['calm_severe']
    precs.append(precision_score(fold_val['is_true_fault'], pred, zero_division=0))
    recs.append(recall_score(fold_val['is_true_fault'], pred, zero_division=0))
    f1s.append(f1_score(fold_val['is_true_fault'], pred, zero_division=0))
print(f"  Support: {train['calm_severe'].sum()} rows (Pos: {train[train['calm_severe']==1]['is_true_fault'].sum()})")
print(f"  CV Precision: {np.mean(precs):.4f}, Recall: {np.mean(recs):.4f}, F1: {np.mean(f1s):.4f}")

# Candidate Rule C: Extreme physical reading (outside [q05, q95])
is_extreme_list = []
for m, r in zip(train['module'], train['reading']):
    if pd.isna(r) or m not in sensor_rules:
        is_extreme_list.append(0)
    else:
        is_extreme_list.append(int(r < sensor_rules[m]['low_b'] or r > sensor_rules[m]['high_b']))
train['sensor_extreme'] = is_extreme_list
print("\nRule C: sensor reading outside [q05, q95] -> is_true_fault = 1")
precs, recs, f1s = [], [], []
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    fold_val = train.iloc[val_i]
    pred = fold_val['sensor_extreme']
    precs.append(precision_score(fold_val['is_true_fault'], pred, zero_division=0))
    recs.append(recall_score(fold_val['is_true_fault'], pred, zero_division=0))
    f1s.append(f1_score(fold_val['is_true_fault'], pred, zero_division=0))
print(f"  Support: {train['sensor_extreme'].sum()} rows (Pos: {train[train['sensor_extreme']==1]['is_true_fault'].sum()})")
print(f"  CV Precision: {np.mean(precs):.4f}, Recall: {np.mean(recs):.4f}, F1: {np.mean(f1s):.4f}")

# --- 6. Train vs Val Distribution Shift Audit ---
print("\n--- 6. TRAIN vs VAL DISTRIBUTION SHIFT AUDIT ---")
for num_c in ['reading', 'delay_ms', 'retry_count']:
    s1 = train[num_c].dropna()
    s2 = val[num_c].dropna()
    ks_stat, p_val = ks_2samp(s1, s2)
    print(f"KS-test on {num_c:<12}: stat={ks_stat:.4f}, p-value={p_val:.4f} -> {'NO SIGNIFICANT SHIFT' if p_val > 0.05 else 'POSSIBLE SHIFT'}")

for cat_c in ['module', 'status', 'mode', 'node']:
    t_c = train[cat_c].value_counts()
    v_c = val[cat_c].value_counts()
    ct = pd.DataFrame({'train': t_c, 'val': v_c}).fillna(0)
    chi2, p_val, dof, ex = chi2_contingency(ct)
    print(f"Chi2-test on {cat_c:<10}: stat={chi2:.4f}, p-value={p_val:.4f} -> {'NO SIGNIFICANT SHIFT' if p_val > 0.05 else 'POSSIBLE SHIFT'}")

# --- 7. Log ID / Row Order Artifact Audit ---
print("\n--- 7. LOG ID / ROW ORDER ARTIFACT AUDIT ---")
train['id_num'] = train['log_id'].apply(lambda x: int(x[1:]))
id_corr = train['id_num'].corr(train['is_true_fault'])
print(f"Correlation between log_id integer and is_true_fault: {id_corr:.4f}")
binned_id = pd.qcut(train['id_num'], q=5)
print("Fault rate by log_id quintiles:")
print(train.groupby(binned_id)['is_true_fault'].agg(['count', 'mean']))
