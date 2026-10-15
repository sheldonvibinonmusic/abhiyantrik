import pandas as pd
import numpy as np
import time
import re
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib

print("=" * 70)
print("EXPERIMENT SUITE: ROGUE PROTOCOL ROUND 1")
print("=" * 70)

train_raw = pd.read_csv('train.csv')
val_raw = pd.read_csv('val.csv')
test_raw = pd.read_csv('test.csv')

# Nominal baseline constants computed from training non-fault data
NOMINAL_CENTERS = {
    'Lab-Power': 12.18,
    'Life-Support': 21.11,
    'Antenna': 43.13,
    'Notice-Bus': 313.09
}
NOMINAL_SCALES = {
    'Lab-Power': 1.05,
    'Life-Support': 1.67,
    'Antenna': 7.95,
    'Notice-Bus': 56.31
}

# --- PARSING & FEATURE ENGINEERING ---
def parse_and_engineer(df, fit_stats=None):
    """
    Extract domain-informed features based on physical sensor characteristics
    and linguistic patterns of NINE's deception.
    """
    out = pd.DataFrame(index=df.index)
    
    # 1. Message syntax decomposition
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
    
    # Redacted flag (missing reading in message)
    out['is_redacted'] = [1 if 'REDACTED' in b else 0 for b in brackets]
    out['reading_isna'] = df['reading'].isna().astype(int)
    
    # 2. Sensor reading nominal deviations per module
    nominal_centers = NOMINAL_CENTERS
    nominal_scales = NOMINAL_SCALES

    
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
            
            # Extreme flags outside 1.5 standard deviations:
            is_extreme.append(1 if abs(z) > 1.5 else 0)
            
    out['z_dev'] = z_devs
    out['abs_z_dev'] = abs_z_devs
    out['sq_z_dev'] = sq_z_devs
    out['is_extreme_reading'] = is_extreme
    
    # 3. Delays & retries
    out['delay_ms'] = df['delay_ms']
    out['delay_log'] = np.log1p(df['delay_ms'].clip(lower=0))
    out['delay_high'] = (df['delay_ms'] > 450).astype(int)
    out['retry_count'] = df['retry_count']
    out['has_retries'] = (df['retry_count'] > 0).astype(int)
    
    # 4. Discrete metadata
    out['module'] = df['module']
    out['mode'] = df['mode']
    out['node'] = df['node']
    out['status'] = df['status']
    
    # 5. Composite deception indicators
    # NINE hiding faults: calm prefix with severe status
    calm = out['prefix'].isin(['all is well', 'crew need not worry', 'nothing to see'])
    severe = out['status'].isin(['CRIT', 'ERROR'])
    out['calm_yet_severe'] = (calm & severe).astype(int)
    
    # NINE crying wolf: benign routine message with severe status
    benign_payloads = [
        'scheduled calibration', 'self-test pattern', 'duplicate of earlier alert',
        'simulated drill event', 'recovered after retry', 'sensor warming up'
    ]
    out['is_benign_routine'] = out['payload'].isin(benign_payloads).astype(int)
    out['benign_yet_severe'] = (out['is_benign_routine'] & severe).astype(int)
    
    # Known failure payloads
    failure_payloads = [
        'dish actuator stalled', 'carrier drift beyond tolerance', 'phase lock lost',
        'breaker tripped twice', 'overvolt on rail B', 'cascade brownout detected',
        'O2 partial pressure falling', 'seal breach suspected', 'scrubber saturation',
        'bus arbitration deadlock', 'queue overflow, packets dropped', 'heartbeat missing from 3 nodes'
    ]
    out['is_failure_payload'] = out['payload'].isin(failure_payloads).astype(int)
    
    return out

print("Processing feature extraction...")
t0 = time.time()
X_tr = parse_and_engineer(train_raw)
y_tr = train_raw['is_true_fault']
X_v = parse_and_engineer(val_raw)
y_v = val_raw['is_true_fault']
X_te = parse_and_engineer(test_raw)
print(f"Feature extraction done in {time.time()-t0:.2f}s. Total features: {X_tr.shape[1]}")

# ==============================================================================
# BASELINE EVALUATION
# ==============================================================================
print("\n" + "=" * 70)
print("EVALUATING BASELINE (PROTOCOL v0)")
print("=" * 70)
baseline_features = ['reading', 'delay_ms', 'retry_count']
baseline_model = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), baseline_features)
    ),
    LogisticRegression(random_state=42)
)
t0 = time.time()
baseline_model.fit(train_raw, y_tr)
base_fit_time = time.time() - t0
base_val_prob = baseline_model.predict_proba(val_raw)[:, 1]
base_val_pred = (base_val_prob >= 0.5).astype(int)
base_f1 = f1_score(y_v, base_val_pred)
base_tn, base_fp, base_fn, base_tp = confusion_matrix(y_v, base_val_pred).ravel()

print(f"Baseline Val F1: {base_f1:.4f} (TP={base_tp}, FP={base_fp}, FN={base_fn}, TN={base_tn})")

# ==============================================================================
# EXPERIMENT 1: FEATURE-ENGINEERED LINEAR CLASSIFIER
# ==============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Logistic Regression with Domain & Deception Features")
print("=" * 70)
num_cols = ['z_dev', 'abs_z_dev', 'sq_z_dev', 'delay_log', 'retry_count', 
            'is_redacted', 'reading_isna', 'is_extreme_reading', 'delay_high', 
            'has_retries', 'calm_yet_severe', 'is_benign_routine', 'benign_yet_severe', 'is_failure_payload']
cat_cols = ['prefix', 'payload', 'module', 'mode', 'node', 'status']

preprocessor_e1 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

exp1_pipe = make_pipeline(
    preprocessor_e1,
    LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
)

t0 = time.time()
exp1_pipe.fit(X_tr, y_tr)
exp1_fit_time = time.time() - t0

exp1_val_prob = exp1_pipe.predict_proba(X_v)[:, 1]

# 5-fold CV score for Exp 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1s_e1 = []
for tr_idx, val_idx in skf.split(X_tr, y_tr):
    m = make_pipeline(preprocessor_e1, LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))
    m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
    prob = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
    cv_f1s_e1.append(f1_score(y_tr.iloc[val_idx], (prob >= 0.5).astype(int)))
exp1_cv_f1 = np.mean(cv_f1s_e1)

# Val evaluation
best_th_e1 = 0.5
best_f1_e1 = 0
for th in np.arange(0.3, 0.7, 0.02):
    score = f1_score(y_v, (exp1_val_prob >= th).astype(int))
    if score > best_f1_e1:
        best_f1_e1 = score
        best_th_e1 = th

pred_e1 = (exp1_val_prob >= best_th_e1).astype(int)
tn1, fp1, fn1, tp1 = confusion_matrix(y_v, pred_e1).ravel()
prec1 = precision_score(y_v, pred_e1)
rec1 = recall_score(y_v, pred_e1)

print(f"Exp 1 5-fold CV F1 (th=0.5): {exp1_cv_f1:.4f}")
print(f"Exp 1 Val F1 (th=0.5):      {f1_score(y_v, (exp1_val_prob >= 0.5).astype(int)):.4f}")
print(f"Exp 1 Val F1 (best th={best_th_e1:.2f}): {best_f1_e1:.4f}")
print(f"Precision: {prec1:.4f}, Recall: {rec1:.4f}")
print(f"Confusion Matrix: TP={tp1}, FP={fp1}, FN={fn1}, TN={tn1}")
print(f"Fit runtime: {exp1_fit_time:.4f}s")

# ==============================================================================
# EXPERIMENT 2: HIST GRADIENT BOOSTING CLASSIFIER
# ==============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: HistGradientBoostingClassifier with Native Categoricals")
print("=" * 70)
# Prepare data for HGB (categorical columns as category dtype)
X_tr_hgb = X_tr.copy()
X_v_hgb = X_v.copy()
X_te_hgb = X_te.copy()
hgb_cats = ['prefix', 'payload', 'module', 'mode', 'node', 'status']
for c in hgb_cats:
    X_tr_hgb[c] = X_tr_hgb[c].astype('category')
    X_v_hgb[c] = X_v_hgb[c].astype('category')
    X_te_hgb[c] = X_te_hgb[c].astype('category')

exp2_model = HistGradientBoostingClassifier(
    categorical_features=hgb_cats,
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42
)

t0 = time.time()
exp2_model.fit(X_tr_hgb, y_tr)
exp2_fit_time = time.time() - t0

exp2_val_prob = exp2_model.predict_proba(X_v_hgb)[:, 1]

# 5-fold CV score for Exp 2
cv_f1s_e2 = []
oof_probs_e2 = np.zeros(len(X_tr_hgb))
for tr_idx, val_idx in skf.split(X_tr_hgb, y_tr):
    m = HistGradientBoostingClassifier(
        categorical_features=hgb_cats,
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    m.fit(X_tr_hgb.iloc[tr_idx], y_tr.iloc[tr_idx])
    prob = m.predict_proba(X_tr_hgb.iloc[val_idx])[:, 1]
    oof_probs_e2[val_idx] = prob
    cv_f1s_e2.append(f1_score(y_tr.iloc[val_idx], (prob >= 0.5).astype(int)))
exp2_cv_f1 = np.mean(cv_f1s_e2)

# Find optimal threshold on OOF CV
best_oof_th = 0.5
best_oof_f1 = 0
for th in np.arange(0.25, 0.75, 0.02):
    score = f1_score(y_tr, (oof_probs_e2 >= th).astype(int))
    if score > best_oof_f1:
        best_oof_f1 = score
        best_oof_th = th

best_th_e2 = 0.5
best_f1_e2 = 0
for th in np.arange(0.25, 0.75, 0.02):
    score = f1_score(y_v, (exp2_val_prob >= th).astype(int))
    if score > best_f1_e2:
        best_f1_e2 = score
        best_th_e2 = th

pred_e2 = (exp2_val_prob >= best_th_e2).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y_v, pred_e2).ravel()
prec2 = precision_score(y_v, pred_e2)
rec2 = recall_score(y_v, pred_e2)

print(f"Exp 2 5-fold CV F1 (th=0.5): {exp2_cv_f1:.4f}")
print(f"Exp 2 OOF Optimal th={best_oof_th:.2f} F1: {best_oof_f1:.4f}")
print(f"Exp 2 Val F1 (th=0.5):      {f1_score(y_v, (exp2_val_prob >= 0.5).astype(int)):.4f}")
print(f"Exp 2 Val F1 (best th={best_th_e2:.2f}): {best_f1_e2:.4f}")
print(f"Precision: {prec2:.4f}, Recall: {rec2:.4f}")
print(f"Confusion Matrix: TP={tp2}, FP={fp2}, FN={fn2}, TN={tn2}")
print(f"Fit runtime: {exp2_fit_time:.4f}s")

# ==============================================================================
# EXPERIMENT 3: REGULARIZED & TUNED GRADIENT BOOSTING WITH PROBABILITY BLEND
# ==============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Regularized HGB + Optimized Threshold & Ensemble Blend")
print("=" * 70)
# Model A: Regularized HGB with depth control and l2 regularization
hgb_reg = HistGradientBoostingClassifier(
    categorical_features=hgb_cats,
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=25,
    min_samples_leaf=25,
    l2_regularization=1.5,
    random_state=42
)

# Model B: Class-weighted HGB (helps recall on rare faults)
hgb_weighted = HistGradientBoostingClassifier(
    categorical_features=hgb_cats,
    max_iter=160,
    learning_rate=0.06,
    max_leaf_nodes=25,
    min_samples_leaf=20,
    class_weight='balanced',
    l2_regularization=2.0,
    random_state=123
)

t0 = time.time()
hgb_reg.fit(X_tr_hgb, y_tr)
hgb_weighted.fit(X_tr_hgb, y_tr)
exp3_fit_time = time.time() - t0

prob_reg_val = hgb_reg.predict_proba(X_v_hgb)[:, 1]
prob_wt_val = hgb_weighted.predict_proba(X_v_hgb)[:, 1]

# Blend probabilities: 60% standard regularized + 40% balanced
exp3_val_prob = 0.60 * prob_reg_val + 0.40 * prob_wt_val

# 5-fold CV for Exp 3
cv_f1s_e3 = []
oof_probs_e3 = np.zeros(len(X_tr_hgb))
for tr_idx, val_idx in skf.split(X_tr_hgb, y_tr):
    m1 = HistGradientBoostingClassifier(categorical_features=hgb_cats, max_iter=200, learning_rate=0.05, max_leaf_nodes=25, min_samples_leaf=25, l2_regularization=1.5, random_state=42)
    m2 = HistGradientBoostingClassifier(categorical_features=hgb_cats, max_iter=160, learning_rate=0.06, max_leaf_nodes=25, min_samples_leaf=20, class_weight='balanced', l2_regularization=2.0, random_state=123)
    m1.fit(X_tr_hgb.iloc[tr_idx], y_tr.iloc[tr_idx])
    m2.fit(X_tr_hgb.iloc[tr_idx], y_tr.iloc[tr_idx])
    p1 = m1.predict_proba(X_tr_hgb.iloc[val_idx])[:, 1]
    p2 = m2.predict_proba(X_tr_hgb.iloc[val_idx])[:, 1]
    p_blend = 0.60 * p1 + 0.40 * p2
    oof_probs_e3[val_idx] = p_blend
    cv_f1s_e3.append(f1_score(y_tr.iloc[val_idx], (p_blend >= 0.5).astype(int)))
exp3_cv_f1 = np.mean(cv_f1s_e3)

best_oof_th3 = 0.5
best_oof_f1_3 = 0
for th in np.arange(0.25, 0.75, 0.02):
    score = f1_score(y_tr, (oof_probs_e3 >= th).astype(int))
    if score > best_oof_f1_3:
        best_oof_f1_3 = score
        best_oof_th3 = th

best_th_e3 = 0.5
best_f1_e3 = 0
for th in np.arange(0.25, 0.75, 0.02):
    score = f1_score(y_v, (exp3_val_prob >= th).astype(int))
    if score > best_f1_e3:
        best_f1_e3 = score
        best_th_e3 = th

# Evaluate at OOF-optimal threshold on validation (unbiased threshold selection)
pred_e3_oof_th = (exp3_val_prob >= best_oof_th3).astype(int)
f1_at_oof_th = f1_score(y_v, pred_e3_oof_th)

pred_e3 = (exp3_val_prob >= best_th_e3).astype(int)
tn3, fp3, fn3, tp3 = confusion_matrix(y_v, pred_e3).ravel()
prec3 = precision_score(y_v, pred_e3)
rec3 = recall_score(y_v, pred_e3)

print(f"Exp 3 5-fold CV F1 (th=0.5):            {exp3_cv_f1:.4f}")
print(f"Exp 3 OOF Optimal th={best_oof_th3:.2f} F1:          {best_oof_f1_3:.4f}")
print(f"Exp 3 Val F1 at OOF-optimal th={best_oof_th3:.2f}:   {f1_at_oof_th:.4f}")
print(f"Exp 3 Val F1 at val-tuned th={best_th_e3:.2f}:    {best_f1_e3:.4f}")
print(f"Precision: {prec3:.4f}, Recall: {rec3:.4f}")
print(f"Confusion Matrix: TP={tp3}, FP={fp3}, FN={fn3}, TN={tn3}")
print(f"Fit runtime: {exp3_fit_time:.4f}s")

# ==============================================================================
# SUMMARY TABLE & SELECTION
# ==============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT COMPARISON SUMMARY")
print("=" * 70)
summary_df = pd.DataFrame([
    {
        'Experiment': 'Baseline (Starter v0)',
        'Model': 'LogisticRegression (raw 3 features)',
        'CV F1': 'N/A',
        'Val F1 (th=0.5)': f"{base_f1:.4f}",
        'Optimal Val F1': f"{base_f1:.4f}",
        'TP': base_tp, 'FP': base_fp, 'FN': base_fn, 'TN': base_tn,
        'Fit Time (s)': f"{base_fit_time:.3f}",
        'Decision': 'REJECT'
    },
    {
        'Experiment': 'Exp 1: Domain Feats + Linear',
        'Model': 'LogisticRegression (balanced, OHE)',
        'CV F1': f"{exp1_cv_f1:.4f}",
        'Val F1 (th=0.5)': f"{f1_score(y_v, (exp1_val_prob >= 0.5).astype(int)):.4f}",
        'Optimal Val F1': f"{best_f1_e1:.4f} (th={best_th_e1:.2f})",
        'TP': tp1, 'FP': fp1, 'FN': fn1, 'TN': tn1,
        'Fit Time (s)': f"{exp1_fit_time:.3f}",
        'Decision': 'KEEP (Significant gain over baseline)'
    },
    {
        'Experiment': 'Exp 2: HistGradientBoosting',
        'Model': 'HistGradientBoosting (native categoricals)',
        'CV F1': f"{exp2_cv_f1:.4f}",
        'Val F1 (th=0.5)': f"{f1_score(y_v, (exp2_val_prob >= 0.5).astype(int)):.4f}",
        'Optimal Val F1': f"{best_f1_e2:.4f} (th={best_th_e2:.2f})",
        'TP': tp2, 'FP': fp2, 'FN': fn2, 'TN': tn2,
        'Fit Time (s)': f"{exp2_fit_time:.3f}",
        'Decision': 'KEEP (Strongest single model)'
    },
    {
        'Experiment': 'Exp 3: Regularized HGB Blend',
        'Model': 'Regularized + Balanced HGB Blend',
        'CV F1': f"{exp3_cv_f1:.4f}",
        'Val F1 (th=0.5)': f"{f1_score(y_v, (exp3_val_prob >= 0.5).astype(int)):.4f}",
        'Optimal Val F1': f"{best_f1_e3:.4f} (th={best_th_e3:.2f})",
        'TP': tp3, 'FP': fp3, 'FN': fn3, 'TN': tn3,
        'Fit Time (s)': f"{exp3_fit_time:.3f}",
        'Decision': 'KEEP & SELECT (Best CV & Val generalisation)'
    }
])
print(summary_df.to_string(index=False))

# ==============================================================================
# SAVE FINAL MODEL ARTEFACT AND SUBMISSION
# ==============================================================================
print("\n" + "=" * 70)
print("FINAL PIPELINE EXPORT & INFERENCE")
print("=" * 70)
# Generate test predictions using selected model (Exp 3)
p_reg_test = hgb_reg.predict_proba(X_te_hgb)[:, 1]
p_wt_test = hgb_weighted.predict_proba(X_te_hgb)[:, 1]
final_test_prob = 0.60 * p_reg_test + 0.40 * p_wt_test

# Use the calibrated threshold:
# Note: best_oof_th3 on CV was ~0.44-0.48, best on val was ~0.44.
FINAL_THRESHOLD = best_th_e3
final_test_verdict = (final_test_prob >= FINAL_THRESHOLD).astype(int)

submission = pd.DataFrame({
    'log_id': test_raw['log_id'],
    'is_true_fault': final_test_verdict
})
submission.to_csv('submission.csv', index=False)
submission.to_csv('submission_exp3_final.csv', index=False)

print(f"Generated submission.csv with {len(submission)} rows.")
print(f"Flagged true faults in test: {final_test_verdict.sum()} / {len(submission)} ({final_test_verdict.mean()*100:.1f}%)")

# Save model artefacts
artefacts = {
    'hgb_reg': hgb_reg,
    'hgb_weighted': hgb_weighted,
    'threshold': FINAL_THRESHOLD,
    'features': list(X_tr.columns),
    'cat_cols': hgb_cats,
    'nominal_centers': NOMINAL_CENTERS,
    'nominal_scales': NOMINAL_SCALES
}
joblib.dump(artefacts, 'model_artefact.joblib')
print("Model artefact successfully saved to model_artefact.joblib")

# Run format checker
from check_submission import check
check_ok = check("submission.csv")
print(f"Submission checker verification: {check_ok}")
