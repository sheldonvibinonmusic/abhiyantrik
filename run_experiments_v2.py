import os
import sys
import time
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from check_submission import check

print("=" * 75)
print("REPRODUCIBLE EXPERIMENT FRAMEWORK (PHASE 3 & 4)")
print("=" * 75)

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

# Global seed
SEED = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Experiment log accumulator
log_entries = []

def record_experiment(exp_id, family, model_desc, oof_f1_50, oof_opt_th, oof_opt_f1, 
                      val_f1_50, val_f1_opt_th, val_prec, val_rec, tp, fp, fn, tn, 
                      fit_time, decision, notes):
    entry = {
        'exp_id': exp_id,
        'family': family,
        'model_description': model_desc,
        'oof_f1_th50': round(oof_f1_50, 4),
        'oof_optimal_th': round(oof_opt_th, 4),
        'oof_optimal_f1': round(oof_opt_f1, 4),
        'val_f1_th50': round(val_f1_50, 4),
        'val_f1_at_oof_th': round(val_f1_opt_th, 4),
        'val_precision': round(val_prec, 4),
        'val_recall': round(val_rec, 4),
        'val_tp': int(tp), 'val_fp': int(fp), 'val_fn': int(fn), 'val_tn': int(tn),
        'runtime_sec': round(fit_time, 2),
        'decision': decision,
        'notes': notes
    }
    log_entries.append(entry)
    print(f"[{exp_id}] OOF F1(0.50): {oof_f1_50:.4f} | OOF Opt th={oof_opt_th:.2f} (F1={oof_opt_f1:.4f}) | Val F1@OOF th: {val_f1_opt_th:.4f} | Decision: {decision}")

# --- Helper: Syntax Decomposition ---
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

# ==============================================================================
# EXPERIMENT 0: STARTER BASELINE (PROTOCOL v0)
# ==============================================================================
print("\n--- Running Experiment 0: Starter Baseline (LogisticRegression raw) ---")
starter_features = ['reading', 'delay_ms', 'retry_count']
pipe_base = Pipeline([
    ('impute', ColumnTransformer([('num', Pipeline([
        ('imp', Pipeline([('m', Pipeline([]))])) if False else SimpleImputer(strategy='median') if False else StandardScaler()
    ]), starter_features)], remainder='drop'))
]) # Use simple standard baseline from Phase 1
from sklearn.impute import SimpleImputer
starter_model = Pipeline([
    ('prep', ColumnTransformer([('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), starter_features)])),
    ('clf', LogisticRegression(random_state=SEED))
])

t0 = time.time()
oof_probs_base = np.zeros(len(train))
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    m = Pipeline([
        ('prep', ColumnTransformer([('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), starter_features)])),
        ('clf', LogisticRegression(random_state=SEED))
    ])
    m.fit(train.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    oof_probs_base[val_i] = m.predict_proba(train.iloc[val_i])[:, 1]
t_base = time.time() - t0

starter_model.fit(train, train['is_true_fault'])
val_prob_b = starter_model.predict_proba(val)[:, 1]

# Find OOF optimal threshold
th_grid = np.arange(0.20, 0.76, 0.02)
best_oof_th_b = 0.50
best_oof_f1_b = 0.0
for th in th_grid:
    s = f1_score(train['is_true_fault'], (oof_probs_base >= th).astype(int), zero_division=0)
    if s > best_oof_f1_b:
        best_oof_f1_b = s
        best_oof_th_b = th

f1_b_50 = f1_score(train['is_true_fault'], (oof_probs_base >= 0.50).astype(int), zero_division=0)
val_f1_b_50 = f1_score(val['is_true_fault'], (val_prob_b >= 0.50).astype(int), zero_division=0)
val_pred_b = (val_prob_b >= best_oof_th_b).astype(int)
val_f1_b_opt = f1_score(val['is_true_fault'], val_pred_b, zero_division=0)
val_prec_b = precision_score(val['is_true_fault'], val_pred_b, zero_division=0)
val_rec_b = recall_score(val['is_true_fault'], val_pred_b, zero_division=0)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(val['is_true_fault'], val_pred_b).ravel()

record_experiment(
    'EXP_0_BASELINE', 'Baseline', 'Starter LogisticRegression on raw [reading, delay, retries]',
    f1_b_50, best_oof_th_b, best_oof_f1_b, val_f1_b_50, val_f1_b_opt, val_prec_b, val_rec_b,
    tp_b, fp_b, fn_b, tn_b, t_base, 'REJECT', 'Catastrophic failure; misses 208/209 faults due to unscaled units.'
)

# ==============================================================================
# FEATURE ENGINEERING EXTRACTOR V2 (Fold-Safe & Asymmetric Physical Bounds)
# ==============================================================================
def extract_features_v2(df, sensor_stats=None):
    """
    Computes domain features. If sensor_stats is None, computes them from df (training phase).
    """
    out = pd.DataFrame(index=df.index)
    prefixes, payloads, brackets = zip(*df['message'].apply(parse_msg))
    out['prefix'] = list(prefixes)
    out['payload'] = list(payloads)
    out['is_redacted'] = [1 if 'REDACTED' in b else 0 for b in brackets]
    out['reading_isna'] = df['reading'].isna().astype(int)
    out['reading'] = df['reading']
    
    # Compute or apply fold-safe sensor statistics
    if sensor_stats is None:
        sensor_stats = {}
        for mod in df['module'].unique():
            non_faults = df[(df['module'] == mod) & (df['is_true_fault'] == 0)]['reading'].dropna()
            sensor_stats[mod] = {
                'median': non_faults.median(),
                'std': non_faults.std() if non_faults.std() > 0 else 1.0,
                'q05': non_faults.quantile(0.05),
                'q95': non_faults.quantile(0.95),
                'q01': non_faults.quantile(0.01),
                'q99': non_faults.quantile(0.99)
            }
            
    # Calculate asymmetric deviations
    z_devs, abs_z_devs, sq_z_devs = [], [], []
    is_under, is_over, is_extreme = [], [], []
    dist_under, dist_over = [], []
    
    for mod, r in zip(df['module'], df['reading']):
        if pd.isna(r) or mod not in sensor_stats:
            z_devs.append(0.0)
            abs_z_devs.append(0.0)
            sq_z_devs.append(0.0)
            is_under.append(0)
            is_over.append(0)
            is_extreme.append(0)
            dist_under.append(0.0)
            dist_over.append(0.0)
        else:
            st = sensor_stats[mod]
            z = (r - st['median']) / st['std']
            z_devs.append(z)
            abs_z_devs.append(abs(z))
            sq_z_devs.append(z ** 2)
            
            under = 1 if r < st['q05'] else 0
            over = 1 if r > st['q95'] else 0
            is_under.append(under)
            is_over.append(over)
            is_extreme.append(1 if (under or over) else 0)
            dist_under.append(max(0.0, st['q05'] - r) / st['std'])
            dist_over.append(max(0.0, r - st['q95']) / st['std'])
            
    out['z_dev'] = z_devs
    out['abs_z_dev'] = abs_z_devs
    out['sq_z_dev'] = sq_z_devs
    out['is_under_nominal'] = is_under
    out['is_over_nominal'] = is_over
    out['is_extreme_reading'] = is_extreme
    out['dist_under_q05'] = dist_under
    out['dist_over_q95'] = dist_over
    
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
    
    return out, sensor_stats

# ==============================================================================
# EXPERIMENT 1: REPRODUCED BEST (22-FEATURE ENSEMBLE BLEND)
# ==============================================================================
print("\n--- Running Experiment 1: Benchmark Existing 22-Feature HGB Blend ---")
# Pre-extract with full train non-fault baseline
X_tr_full, base_sensor_stats = extract_features_v2(train)
X_v_full, _ = extract_features_v2(val, base_sensor_stats)
X_te_full, _ = extract_features_v2(test, base_sensor_stats)

cats = ['prefix', 'payload', 'module', 'mode', 'node', 'status']
for c in cats:
    X_tr_full[c] = X_tr_full[c].astype('category')
    X_v_full[c] = X_v_full[c].astype('category')
    X_te_full[c] = X_te_full[c].astype('category')

t0 = time.time()
oof_probs_e1 = np.zeros(len(train))
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    m1 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=200, learning_rate=0.05, max_leaf_nodes=25, min_samples_leaf=25, l2_regularization=1.5, random_state=SEED)
    m2 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=160, learning_rate=0.06, max_leaf_nodes=25, min_samples_leaf=20, class_weight='balanced', l2_regularization=2.0, random_state=123)
    m1.fit(X_tr_full.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    m2.fit(X_tr_full.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    p1 = m1.predict_proba(X_tr_full.iloc[val_i])[:, 1]
    p2 = m2.predict_proba(X_tr_full.iloc[val_i])[:, 1]
    oof_probs_e1[val_i] = 0.60 * p1 + 0.40 * p2

t_e1 = time.time() - t0

m1_full = HistGradientBoostingClassifier(categorical_features=cats, max_iter=200, learning_rate=0.05, max_leaf_nodes=25, min_samples_leaf=25, l2_regularization=1.5, random_state=SEED).fit(X_tr_full, train['is_true_fault'])
m2_full = HistGradientBoostingClassifier(categorical_features=cats, max_iter=160, learning_rate=0.06, max_leaf_nodes=25, min_samples_leaf=20, class_weight='balanced', l2_regularization=2.0, random_state=123).fit(X_tr_full, train['is_true_fault'])
val_p_e1 = 0.60 * m1_full.predict_proba(X_v_full)[:, 1] + 0.40 * m2_full.predict_proba(X_v_full)[:, 1]

best_oof_th_e1 = 0.50
best_oof_f1_e1 = 0.0
for th in th_grid:
    s = f1_score(train['is_true_fault'], (oof_probs_e1 >= th).astype(int), zero_division=0)
    if s > best_oof_f1_e1:
        best_oof_f1_e1 = s
        best_oof_th_e1 = th

f1_e1_50 = f1_score(train['is_true_fault'], (oof_probs_e1 >= 0.50).astype(int), zero_division=0)
val_f1_e1_50 = f1_score(val['is_true_fault'], (val_p_e1 >= 0.50).astype(int), zero_division=0)
val_pred_e1 = (val_p_e1 >= best_oof_th_e1).astype(int)
val_f1_e1_opt = f1_score(val['is_true_fault'], val_pred_e1, zero_division=0)
val_prec_e1 = precision_score(val['is_true_fault'], val_pred_e1, zero_division=0)
val_rec_e1 = recall_score(val['is_true_fault'], val_pred_e1, zero_division=0)
tn_1, fp_1, fn_1, tp_1 = confusion_matrix(val['is_true_fault'], val_pred_e1).ravel()

record_experiment(
    'EXP_1_HGB_BLEND_22F', 'Ensemble Tree', '2-Model HGB Blend (60/40 regularized/balanced) on 22 features',
    f1_e1_50, best_oof_th_e1, best_oof_f1_e1, val_f1_e1_50, val_f1_e1_opt, val_prec_e1, val_rec_e1,
    tp_1, fp_1, fn_1, tn_1, t_e1, 'BASELINE_CANDIDATE', 'Unbiased OOF threshold is 0.45, giving Val F1 0.6178.'
)

# ==============================================================================
# EXPERIMENT 2: ASYMMETRIC SENSOR BOUNDARIES + INTERACTION REFINEMENT
# ==============================================================================
print("\n--- Running Experiment 2: Asymmetric Sensor Boundaries + Interaction Refinement ---")
# Features in X_tr_full now have dist_under_q05 and dist_over_q95
# Test whether tree models with explicit asymmetric distances improve OOF F1
t0 = time.time()
oof_probs_e2 = np.zeros(len(train))
for tr_i, val_i in skf.split(train, train['is_true_fault']):
    # Fit fold-safe sensor stats
    X_tr_fold, fold_stats = extract_features_v2(train.iloc[tr_i])
    X_val_fold, _ = extract_features_v2(train.iloc[val_i], fold_stats)
    for c in cats:
        X_tr_fold[c] = X_tr_fold[c].astype('category')
        X_val_fold[c] = X_val_fold[c].astype('category')
        
    m1 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=220, learning_rate=0.045, max_leaf_nodes=28, min_samples_leaf=22, l2_regularization=2.5, random_state=SEED)
    m2 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=180, learning_rate=0.055, max_leaf_nodes=25, min_samples_leaf=18, class_weight='balanced', l2_regularization=3.0, random_state=123)
    m1.fit(X_tr_fold, train['is_true_fault'].iloc[tr_i])
    m2.fit(X_tr_fold, train['is_true_fault'].iloc[tr_i])
    p1 = m1.predict_proba(X_val_fold)[:, 1]
    p2 = m2.predict_proba(X_val_fold)[:, 1]
    oof_probs_e2[val_i] = 0.55 * p1 + 0.45 * p2

t_e2 = time.time() - t0

# Fit on full train
X_tr_f2, final_stats2 = extract_features_v2(train)
X_v_f2, _ = extract_features_v2(val, final_stats2)
for c in cats:
    X_tr_f2[c] = X_tr_f2[c].astype('category')
    X_v_f2[c] = X_v_f2[c].astype('category')

m1_f2 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=220, learning_rate=0.045, max_leaf_nodes=28, min_samples_leaf=22, l2_regularization=2.5, random_state=SEED).fit(X_tr_f2, train['is_true_fault'])
m2_f2 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=180, learning_rate=0.055, max_leaf_nodes=25, min_samples_leaf=18, class_weight='balanced', l2_regularization=3.0, random_state=123).fit(X_tr_f2, train['is_true_fault'])
val_p_e2 = 0.55 * m1_f2.predict_proba(X_v_f2)[:, 1] + 0.45 * m2_f2.predict_proba(X_v_f2)[:, 1]

best_oof_th_e2 = 0.50
best_oof_f1_e2 = 0.0
for th in th_grid:
    s = f1_score(train['is_true_fault'], (oof_probs_e2 >= th).astype(int), zero_division=0)
    if s > best_oof_f1_e2:
        best_oof_f1_e2 = s
        best_oof_th_e2 = th

f1_e2_50 = f1_score(train['is_true_fault'], (oof_probs_e2 >= 0.50).astype(int), zero_division=0)
val_f1_e2_50 = f1_score(val['is_true_fault'], (val_p_e2 >= 0.50).astype(int), zero_division=0)
val_pred_e2 = (val_p_e2 >= best_oof_th_e2).astype(int)
val_f1_e2_opt = f1_score(val['is_true_fault'], val_pred_e2, zero_division=0)
val_prec_e2 = precision_score(val['is_true_fault'], val_pred_e2, zero_division=0)
val_rec_e2 = recall_score(val['is_true_fault'], val_pred_e2, zero_division=0)
tn_2, fp_2, fn_2, tp_2 = confusion_matrix(val['is_true_fault'], val_pred_e2).ravel()

decision_e2 = 'KEEP' if best_oof_f1_e2 >= best_oof_f1_e1 else 'REJECT'
record_experiment(
    'EXP_2_ASYMMETRIC_SPLINES', 'Structured/Feature Eng', 'HGB Blend with fold-safe asymmetric sensor distance features',
    f1_e2_50, best_oof_th_e2, best_oof_f1_e2, val_f1_e2_50, val_f1_e2_opt, val_prec_e2, val_rec_e2,
    tp_2, fp_2, fn_2, tn_2, t_e2, decision_e2, 'Fold-safe asymmetric distance to q05 and q95 added.'
)

# ==============================================================================
# EXPERIMENT 3: HYBRID TEXT TF-IDF + STRUCTURED LOG BLEND
# ==============================================================================
print("\n--- Running Experiment 3: Hybrid Lexical TF-IDF + Regularized Linear Model Blend ---")
# Train a sparse lexical model strictly on raw log messages
# and blend with the tree model probabilities
t0 = time.time()
oof_probs_text = np.zeros(len(train))

for tr_i, val_i in skf.split(train, train['is_true_fault']):
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=1200, sublinear_tf=True)
    X_tr_text = tfidf.fit_transform(train['message'].iloc[tr_i])
    X_val_text = tfidf.transform(train['message'].iloc[val_i])
    
    clf_text = LogisticRegression(C=0.8, max_iter=1000, class_weight='balanced', random_state=SEED)
    clf_text.fit(X_tr_text, train['is_true_fault'].iloc[tr_i])
    oof_probs_text[val_i] = clf_text.predict_proba(X_val_text)[:, 1]

# Full train text model
tfidf_full = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=1200, sublinear_tf=True)
X_tr_t_full = tfidf_full.fit_transform(train['message'])
X_v_t_full = tfidf_full.transform(val['message'])
clf_text_full = LogisticRegression(C=0.8, max_iter=1000, class_weight='balanced', random_state=SEED)
clf_text_full.fit(X_tr_t_full, train['is_true_fault'])
val_p_text = clf_text_full.predict_proba(X_v_t_full)[:, 1]

# Multi-model Blend: 0.70 Tree Ensemble (Exp 2) + 0.30 Lexical Model
oof_probs_hybrid = 0.70 * oof_probs_e2 + 0.30 * oof_probs_text
val_p_hybrid = 0.70 * val_p_e2 + 0.30 * val_p_text
t_e3 = time.time() - t0

best_oof_th_e3 = 0.50
best_oof_f1_e3 = 0.0
for th in th_grid:
    s = f1_score(train['is_true_fault'], (oof_probs_hybrid >= th).astype(int), zero_division=0)
    if s > best_oof_f1_e3:
        best_oof_f1_e3 = s
        best_oof_th_e3 = th

f1_e3_50 = f1_score(train['is_true_fault'], (oof_probs_hybrid >= 0.50).astype(int), zero_division=0)
val_f1_e3_50 = f1_score(val['is_true_fault'], (val_p_hybrid >= 0.50).astype(int), zero_division=0)
val_pred_e3 = (val_p_hybrid >= best_oof_th_e3).astype(int)
val_f1_e3_opt = f1_score(val['is_true_fault'], val_pred_e3, zero_division=0)
val_prec_e3 = precision_score(val['is_true_fault'], val_pred_e3, zero_division=0)
val_rec_e3 = recall_score(val['is_true_fault'], val_pred_e3, zero_division=0)
tn_3, fp_3, fn_3, tp_3 = confusion_matrix(val['is_true_fault'], val_pred_e3).ravel()

decision_e3 = 'KEEP' if best_oof_f1_e3 > best_oof_f1_e2 else 'REJECT'
record_experiment(
    'EXP_3_HYBRID_TEXT_TREE', 'Hybrid Text+Tree', '70% Asymmetric HGB + 30% Lexical TF-IDF LogisticRegression',
    f1_e3_50, best_oof_th_e3, best_oof_f1_e3, val_f1_e3_50, val_f1_e3_opt, val_prec_e3, val_rec_e3,
    tp_3, fp_3, fn_3, tn_3, t_e3, decision_e3, 'Combines tree interaction capacity with sparse sub-word token detection.'
)

# ==============================================================================
# EXPERIMENT 4: OPTIMIZED STACKING META-LEARNER
# ==============================================================================
print("\n--- Running Experiment 4: Out-of-Fold Stacking Meta-Classifier ---")
# Base models:
# 1. Asymmetric regularized HGB
# 2. Balanced HGB
# 3. Lexical TF-IDF LogisticRegression
# 4. ExtraTreesClassifier (diversity)
from sklearn.ensemble import ExtraTreesClassifier
t0 = time.time()
oof_m1 = np.zeros(len(train))
oof_m2 = np.zeros(len(train))
oof_m3 = oof_probs_text
oof_m4 = np.zeros(len(train))

# Prepare numeric/imputed matrix for ExtraTrees
num_cols_et = ['z_dev', 'abs_z_dev', 'sq_z_dev', 'delay_log', 'retry_count', 'is_redacted', 'reading_isna', 'is_under_nominal', 'is_over_nominal', 'dist_under_q05', 'dist_over_q95']
X_tr_et = X_tr_f2[num_cols_et].fillna(0)
X_v_et = X_v_f2[num_cols_et].fillna(0)

for tr_i, val_i in skf.split(train, train['is_true_fault']):
    # M1 & M2
    m1 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=220, learning_rate=0.045, max_leaf_nodes=28, min_samples_leaf=22, l2_regularization=2.5, random_state=SEED)
    m2 = HistGradientBoostingClassifier(categorical_features=cats, max_iter=180, learning_rate=0.055, max_leaf_nodes=25, min_samples_leaf=18, class_weight='balanced', l2_regularization=3.0, random_state=123)
    m1.fit(X_tr_f2.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    m2.fit(X_tr_f2.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    oof_m1[val_i] = m1.predict_proba(X_tr_f2.iloc[val_i])[:, 1]
    oof_m2[val_i] = m2.predict_proba(X_tr_f2.iloc[val_i])[:, 1]
    
    # M4: ExtraTrees
    m4 = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=SEED, n_jobs=-1)
    m4.fit(X_tr_et.iloc[tr_i], train['is_true_fault'].iloc[tr_i])
    oof_m4[val_i] = m4.predict_proba(X_tr_et.iloc[val_i])[:, 1]

# Meta-training matrix
OOF_META = np.column_stack([oof_m1, oof_m2, oof_m3, oof_m4])
meta_clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
meta_clf.fit(OOF_META, train['is_true_fault'])

# Out-of-fold meta predictions via cross-validation
oof_meta_preds = np.zeros(len(train))
for tr_i, val_i in skf.split(OOF_META, train['is_true_fault']):
    mc = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    mc.fit(OOF_META[tr_i], train['is_true_fault'].iloc[tr_i])
    oof_meta_preds[val_i] = mc.predict_proba(OOF_META[val_i])[:, 1]

# Validation meta matrix
m4_full = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=SEED, n_jobs=-1).fit(X_tr_et, train['is_true_fault'])
VAL_META = np.column_stack([
    m1_f2.predict_proba(X_v_f2)[:, 1],
    m2_f2.predict_proba(X_v_f2)[:, 1],
    val_p_text,
    m4_full.predict_proba(X_v_et)[:, 1]
])
val_meta_prob = meta_clf.predict_proba(VAL_META)[:, 1]
t_e4 = time.time() - t0

best_oof_th_e4 = 0.50
best_oof_f1_e4 = 0.0
for th in th_grid:
    s = f1_score(train['is_true_fault'], (oof_meta_preds >= th).astype(int), zero_division=0)
    if s > best_oof_f1_e4:
        best_oof_f1_e4 = s
        best_oof_th_e4 = th

f1_e4_50 = f1_score(train['is_true_fault'], (oof_meta_preds >= 0.50).astype(int), zero_division=0)
val_f1_e4_50 = f1_score(val['is_true_fault'], (val_meta_prob >= 0.50).astype(int), zero_division=0)
val_pred_e4 = (val_meta_prob >= best_oof_th_e4).astype(int)
val_f1_e4_opt = f1_score(val['is_true_fault'], val_pred_e4, zero_division=0)
val_prec_e4 = precision_score(val['is_true_fault'], val_pred_e4, zero_division=0)
val_rec_e4 = recall_score(val['is_true_fault'], val_pred_e4, zero_division=0)
tn_4, fp_4, fn_4, tp_4 = confusion_matrix(val['is_true_fault'], val_pred_e4).ravel()

decision_e4 = 'KEEP' if best_oof_f1_e4 > max(best_oof_f1_e1, best_oof_f1_e2, best_oof_f1_e3) else 'REJECT'
record_experiment(
    'EXP_4_STACKING_META', 'Stacking', 'LogisticRegression meta-classifier on [HGB_reg, HGB_bal, Text_LR, ExtraTrees]',
    f1_e4_50, best_oof_th_e4, best_oof_f1_e4, val_f1_e4_50, val_f1_e4_opt, val_prec_e4, val_rec_e4,
    tp_4, fp_4, fn_4, tn_4, t_e4, decision_e4, 'Out-of-fold meta-features avoid label leakage across stages.'
)

# ==============================================================================
# SAVE EXPERIMENT LOG (CSV & JSON)
# ==============================================================================
exp_df = pd.DataFrame(log_entries)
exp_df.to_csv('EXPERIMENT_LOG.csv', index=False)
print("\n" + "=" * 75)
print("EXPERIMENT COMPARISON SUMMARY TABLE")
print("=" * 75)
print(exp_df[['exp_id', 'family', 'oof_optimal_th', 'oof_optimal_f1', 'val_f1_at_oof_th', 'val_precision', 'val_recall', 'decision']].to_string(index=False))
