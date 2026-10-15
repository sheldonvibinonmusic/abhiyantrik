"""
ROGUE PROTOCOL - TRANSFORMER TRAINING & EVALUATION PIPELINE
Executes Candidates A (MiniLM Embeddings + Structured), B (Fine-tuned DistilBERT),
and C (Multimodal Fusion / Hybrid Blend) 100% offline using local GPU.
"""
import os
import sys
import time
import re
import json
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Enforce 100% offline local execution
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("=" * 75)
print(f"TRANSFORMER LOCAL TRAINING PIPELINE ON DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | Free VRAM: {torch.cuda.mem_get_info(0)[0]/(1024**3):.2f} GB")
print("=" * 75)

# 1. Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

print(f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Helper: parse messages for structured features
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

def engineer_structured_features(df, nominal_centers, nominal_scales):
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
    
    cat_cols = ['prefix', 'payload', 'module', 'mode', 'node', 'status']
    for c in cat_cols:
        out[c] = out[c].astype('category')
    return out, cat_cols

# Compute fold-safe base nominal statistics
nominal_centers = train_df.groupby('module')['reading'].median().to_dict()
nominal_scales = {}
for m in nominal_centers:
    sub_r = train_df[train_df['module'] == m]['reading'].dropna()
    iqr = sub_r.quantile(0.75) - sub_r.quantile(0.25)
    nominal_scales[m] = (iqr / 1.349) if iqr > 1e-4 else 1.0

X_train_struct, cat_cols = engineer_structured_features(train_df, nominal_centers, nominal_scales)
X_val_struct, _ = engineer_structured_features(val_df, nominal_centers, nominal_scales)
X_test_struct, _ = engineer_structured_features(test_df, nominal_centers, nominal_scales)

# Clean input strings for text models
def format_input_text(df):
    texts = []
    for _, row in df.iterrows():
        m = str(row['message']) if pd.notna(row['message']) else ""
        if m.startswith('NINE:'):
            m = m[5:].strip()
        mod = str(row.get('module', ''))
        st = str(row.get('status', ''))
        texts.append(f"Module: {mod} | Status: {st} | Message: {m}")
    return texts

train_texts = format_input_text(train_df)
val_texts = format_input_text(val_df)
test_texts = format_input_text(test_df)

# ==============================================================================
# CANDIDATE A: EXTRACT FROZEN MiniLM SENTENCE EMBEDDINGS (100% LOCAL GPU)
# ==============================================================================
print("\n" + "=" * 70)
print("[CANDIDATE A] Extracting all-MiniLM-L6-v2 Embeddings on Local GPU...")
print("=" * 70)

tok_minilm = AutoTokenizer.from_pretrained('models/all-MiniLM-L6-v2', local_files_only=True)
model_minilm = AutoModel.from_pretrained('models/all-MiniLM-L6-v2', local_files_only=True).to(DEVICE)
model_minilm.eval()

def extract_minilm_embeddings(text_list, batch_size=128):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            encoded = tok_minilm(batch, padding=True, truncation=True, max_length=96, return_tensors='pt').to(DEVICE)
            outputs = model_minilm(**encoded)
            # Mean pooling over attention mask
            token_embeddings = outputs.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(mean_pooled)
    return np.vstack(all_embeddings)

t0 = time.time()
emb_train = extract_minilm_embeddings(train_texts)
emb_val = extract_minilm_embeddings(val_texts)
emb_test = extract_minilm_embeddings(test_texts)
t_emb = time.time() - t0
print(f"Extracted MiniLM embeddings in {t_emb:.2f}s (Shape: {emb_train.shape})")

# Combine MiniLM embeddings with structured continuous features
num_cols = ['z_dev', 'abs_z_dev', 'sq_z_dev', 'is_extreme_reading', 'delay_log', 'retry_count', 'is_redacted', 'reading_isna', 'calm_yet_severe', 'is_benign_routine', 'benign_yet_severe', 'is_failure_payload']
X_tr_num = X_train_struct[num_cols].fillna(0).values
X_v_num = X_val_struct[num_cols].fillna(0).values
X_te_num = X_test_struct[num_cols].fillna(0).values

X_tr_dense = np.hstack([emb_train, X_tr_num])
X_v_dense = np.hstack([emb_val, X_v_num])
X_te_dense = np.hstack([emb_test, X_te_num])

print(f"Combined dense feature representation shape: {X_tr_dense.shape}")

# 3-Fold Stratified CV Evaluation of Candidate A
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
th_grid = np.arange(0.20, 0.65, 0.01)

print("\n--- Evaluating Candidate A1: MiniLM + Structured with LogisticRegression ---")
oof_p_a1 = np.zeros(len(train_df))
val_p_a1 = np.zeros(len(val_df))

for tr_idx, val_idx in skf.split(train_df, train_df['is_true_fault']):
    clf_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight='balanced')
    clf_lr.fit(X_tr_dense[tr_idx], train_df['is_true_fault'].iloc[tr_idx])
    oof_p_a1[val_idx] = clf_lr.predict_proba(X_tr_dense[val_idx])[:, 1]
    val_p_a1 += clf_lr.predict_proba(X_v_dense)[:, 1] / 3

best_th_a1 = 0.50
best_f1_a1 = 0.0
for th in th_grid:
    f = f1_score(train_df['is_true_fault'], (oof_p_a1 >= th).astype(int), zero_division=0)
    if f > best_f1_a1:
        best_f1_a1 = f
        best_th_a1 = th

val_f1_a1 = f1_score(val_df['is_true_fault'], (val_p_a1 >= best_th_a1).astype(int), zero_division=0)
val_prec_a1 = precision_score(val_df['is_true_fault'], (val_p_a1 >= best_th_a1).astype(int), zero_division=0)
val_rec_a1 = recall_score(val_df['is_true_fault'], (val_p_a1 >= best_th_a1).astype(int), zero_division=0)
print(f"Candidate A1 (MiniLM + LogReg): OOF Opt th={best_th_a1:.2f} (F1={best_f1_a1:.4f}) | Val F1={val_f1_a1:.4f} (Prec: {val_prec_a1:.4f}, Rec: {val_rec_a1:.4f})")

# ==============================================================================
# CANDIDATE B: FINE-TUNED DISTILBERT CLASSIFIER (100% LOCAL GPU)
# ==============================================================================
print("\n" + "=" * 70)
print("[CANDIDATE B] Fine-tuning DistilBERT Sequence Classifier on Local GPU...")
print("=" * 70)

tok_distil = AutoTokenizer.from_pretrained('models/distilbert-base-uncased', local_files_only=True)

class LogDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = {'text': self.texts[idx]}
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

def collate_fn(batch):
    texts = [x['text'] for x in batch]
    encoded = tok_distil(texts, padding=True, truncation=True, max_length=96, return_tensors='pt')
    if 'label' in batch[0]:
        encoded['labels'] = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    return encoded

train_ds = LogDataset(train_texts, train_df['is_true_fault'].values)
val_ds = LogDataset(val_texts, val_df['is_true_fault'].values)
test_ds = LogDataset(test_texts)

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

# 3-Fold Stratified Fine-tuning of DistilBERT
oof_distil_probs = np.zeros(len(train_df))
val_distil_probs = np.zeros(len(val_df))
test_distil_probs = np.zeros(len(test_df))

t0_distil = time.time()
for fold, (tr_idx, v_idx) in enumerate(skf.split(train_df, train_df['is_true_fault'])):
    print(f"\n--- Training DistilBERT Fold {fold+1}/3 ---")
    tr_sub_ds = LogDataset([train_texts[i] for i in tr_idx], train_df['is_true_fault'].iloc[tr_idx].values)
    va_sub_ds = LogDataset([train_texts[i] for i in v_idx], train_df['is_true_fault'].iloc[v_idx].values)
    
    tr_loader = DataLoader(tr_sub_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_sub_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model_b = AutoModelForSequenceClassification.from_pretrained('models/distilbert-base-uncased', num_labels=2, local_files_only=True).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model_b.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 2
    total_steps = len(tr_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    model_b.train()
    for ep in range(epochs):
        for step, batch in enumerate(tr_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model_b(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model_b(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
    # Fold inference
    model_b.eval()
    va_fold_preds = []
    with torch.no_grad():
        for batch in va_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model_b(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            va_fold_preds.extend(probs)
    oof_distil_probs[v_idx] = va_fold_preds
    
    # Predict on held-out val & test
    val_fold_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model_b(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_fold_preds.extend(probs)
    val_distil_probs += np.array(val_fold_preds) / 3
    
    test_fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model_b(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            test_fold_preds.extend(probs)
    test_distil_probs += np.array(test_fold_preds) / 3

t_distil_total = time.time() - t0_distil
print(f"\nDistilBERT 3-Fold fine-tuning completed in {t_distil_total:.2f}s")

# Evaluate Candidate B
best_th_b = 0.50
best_f1_b = 0.0
for th in th_grid:
    f = f1_score(train_df['is_true_fault'], (oof_distil_probs >= th).astype(int), zero_division=0)
    if f > best_f1_b:
        best_f1_b = f
        best_th_b = th

val_f1_b = f1_score(val_df['is_true_fault'], (val_distil_probs >= best_th_b).astype(int), zero_division=0)
val_prec_b = precision_score(val_df['is_true_fault'], (val_distil_probs >= best_th_b).astype(int), zero_division=0)
val_rec_b = recall_score(val_df['is_true_fault'], (val_distil_probs >= best_th_b).astype(int), zero_division=0)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(val_df['is_true_fault'], (val_distil_probs >= best_th_b).astype(int)).ravel()

print(f"Candidate B (DistilBERT Fine-tuned): OOF Opt th={best_th_b:.2f} (F1={best_f1_b:.4f}) | Val F1={val_f1_b:.4f} (Prec: {val_prec_b:.4f}, Rec: {val_rec_b:.4f})")
print(f"Val Confusion Matrix: TP={tp_b}, FP={fp_b}, FN={fn_b}, TN={tn_b}")

# ==============================================================================
# CANDIDATE C: MULTIMODAL FUSION & HYBRID BLEND
# ==============================================================================
print("\n" + "=" * 70)
print("[CANDIDATE C] Multimodal Hybrid Blending (DistilBERT + MiniLM + Structured HGB)")
print("=" * 70)

# Load existing structured model probabilities
existing_artefact = joblib.load('model_artefact_verified.joblib')
m1_struct = existing_artefact['struct_m1']
m2_struct = existing_artefact['struct_m2']
centers = existing_artefact['nominal_centers']
scales = existing_artefact['nominal_scales']
cats = existing_artefact['cat_cols']

X_tr_s, _ = engineer_structured_features(train_df, centers, scales)
X_v_s, _ = engineer_structured_features(val_df, centers, scales)
X_te_s, _ = engineer_structured_features(test_df, centers, scales)

p_struct_train = 0.60 * m1_struct.predict_proba(X_tr_s)[:, 1] + 0.40 * m2_struct.predict_proba(X_tr_s)[:, 1]
p_struct_val = 0.60 * m1_struct.predict_proba(X_v_s)[:, 1] + 0.40 * m2_struct.predict_proba(X_v_s)[:, 1]
p_struct_test = 0.60 * m1_struct.predict_proba(X_te_s)[:, 1] + 0.40 * m2_struct.predict_proba(X_te_s)[:, 1]

# Search optimal hybrid blend weight: w * p_struct + (1-w) * p_distil
best_w = 0.85
best_th_c = 0.50
best_f1_c = 0.0

for w in np.arange(0.50, 0.95, 0.05):
    p_hybrid_oof = w * p_struct_train + (1.0 - w) * oof_distil_probs
    for th in th_grid:
        f = f1_score(train_df['is_true_fault'], (p_hybrid_oof >= th).astype(int), zero_division=0)
        if f > best_f1_c:
            best_f1_c = f
            best_w = w
            best_th_c = th

p_hybrid_val = best_w * p_struct_val + (1.0 - best_w) * val_distil_probs
p_hybrid_test = best_w * p_struct_test + (1.0 - best_w) * test_distil_probs

val_f1_c = f1_score(val_df['is_true_fault'], (p_hybrid_val >= best_th_c).astype(int), zero_division=0)
val_prec_c = precision_score(val_df['is_true_fault'], (p_hybrid_val >= best_th_c).astype(int), zero_division=0)
val_rec_c = recall_score(val_df['is_true_fault'], (p_hybrid_val >= best_th_c).astype(int), zero_division=0)
tn_c, fp_c, fn_c, tp_c = confusion_matrix(val_df['is_true_fault'], (p_hybrid_val >= best_th_c).astype(int)).ravel()

print(f"\nCandidate C (Multimodal Hybrid: {best_w:.2f} Structured + {1-best_w:.2f} DistilBERT):")
print(f"  OOF Optimal Th: {best_th_c:.2f} -> OOF F1: {best_f1_c:.4f}")
print(f"  Val F1 at OOF Th: {val_f1_c:.4f} | Val Precision: {val_prec_c:.4f} | Val Recall: {val_rec_c:.4f}")
print(f"  Val Confusion Matrix: TP={tp_c}, FP={fp_c}, FN={fn_c}, TN={tn_c}")

# ==============================================================================
# SUMMARY & ARTIFACT PERSISTENCE
# ==============================================================================
print("\n" + "=" * 75)
print("FINAL MODEL BENCHMARK COMPARISON TABLE")
print("=" * 75)
results = [
    {'model': 'Benchmark Baseline (Existing Hybrid)', 'oof_f1': 0.6345, 'val_f1': 0.6011, 'th': 0.50, 'status': 'EXISTING VERIFIED'},
    {'model': 'Candidate A1 (Frozen MiniLM + Structured LogReg)', 'oof_f1': round(best_f1_a1, 4), 'val_f1': round(val_f1_a1, 4), 'th': round(best_th_a1, 2), 'status': 'VERIFIED LOCAL'},
    {'model': 'Candidate B (Fine-tuned DistilBERT 100% Local)', 'oof_f1': round(best_f1_b, 4), 'val_f1': round(val_f1_b, 4), 'th': round(best_th_b, 2), 'status': 'VERIFIED LOCAL'},
    {'model': 'Candidate C (Multimodal Fusion Hybrid)', 'oof_f1': round(best_f1_c, 4), 'val_f1': round(val_f1_c, 4), 'th': round(best_th_c, 2), 'status': 'VERIFIED LOCAL'}
]
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# Save candidate transformer test predictions
preds_c = (p_hybrid_test >= best_th_c).astype(int)
sub_trans = pd.DataFrame({
    'log_id': test_df['log_id'],
    'is_true_fault': preds_c
})
sub_trans.to_csv('submission_transformer_candidate.csv', index=False)
print(f"\nSaved submission_transformer_candidate.csv ({len(sub_trans)} rows, {preds_c.sum()} flagged faults).")

# Save evaluation report to artifacts
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/transformer_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Artifacts saved successfully.")
