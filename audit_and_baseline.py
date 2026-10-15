import pandas as pd
import numpy as np
import time
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from check_submission import check

print("=" * 60)
print("1. LOADING DATASETS")
print("=" * 60)
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Val shape:   {val.shape}")
print(f"Test shape:  {test.shape}")

print("\n" + "=" * 60)
print("2. SCHEMA & TYPES")
print("=" * 60)
print(train.dtypes)

print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)
print("Train missing:")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("\nVal missing:")
print(val.isnull().sum()[val.isnull().sum() > 0])
print("\nTest missing:")
print(test.isnull().sum()[test.isnull().sum() > 0])

print("\n" + "=" * 60)
print("4. TARGET DISTRIBUTION")
print("=" * 60)
print("Train target value counts:")
print(train['is_true_fault'].value_counts(normalize=True))
print("Val target value counts:")
print(val['is_true_fault'].value_counts(normalize=True))

print("\n" + "=" * 60)
print("5. CATEGORICAL FEATURES")
print("=" * 60)
for col in ['module', 'mode', 'node', 'status']:
    if col in train.columns:
        print(f"\n--- {col} ---")
        print(f"Train unique: {train[col].nunique()} -> {train[col].unique()}")
        print(f"Val unique:   {val[col].nunique()} -> {val[col].unique()}")
        print(f"Test unique:  {test[col].nunique()} -> {test[col].unique()}")
        # Check for unseen categories
        unseen_val = set(val[col].dropna()) - set(train[col].dropna())
        unseen_test = set(test[col].dropna()) - set(train[col].dropna())
        if unseen_val:
            print(f"  Unseen in val: {unseen_val}")
        if unseen_test:
            print(f"  Unseen in test: {unseen_test}")

print("\n" + "=" * 60)
print("6. NUMERICAL FEATURES SUMMARY")
print("=" * 60)
num_cols = ['reading', 'delay_ms', 'retry_count']
print("Train stats:")
print(train[num_cols].describe().T)
print("\nVal stats:")
print(val[num_cols].describe().T)
print("\nTest stats:")
print(test[num_cols].describe().T)

print("\n" + "=" * 60)
print("7. SAMPLE MESSAGES BY TARGET")
print("=" * 60)
print("True faults (is_true_fault == 1):")
for msg in train[train['is_true_fault'] == 1]['message'].head(5):
    print("  [1] ", msg)
print("\nFalse alarms / noise (is_true_fault == 0):")
for msg in train[train['is_true_fault'] == 0]['message'].head(5):
    print("  [0] ", msg)

print("\n" + "=" * 60)
print("8. TIMESTAMPS & RECORD ORDER")
print("=" * 60)
print(f"Train time range: {train['timestamp'].min()} to {train['timestamp'].max()}")
print(f"Val time range:   {val['timestamp'].min()} to {val['timestamp'].max()}")
print(f"Test time range:  {test['timestamp'].min()} to {test['timestamp'].max()}")

print("\n" + "=" * 60)
print("9. RUNNING STARTER BASELINE (ACT 2 - ACT 5)")
print("=" * 60)
FEATURES = ['reading', 'delay_ms', 'retry_count']
starter_model = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), FEATURES)
    ),
    LogisticRegression(random_state=42),
)

t0 = time.time()
starter_model.fit(train, train['is_true_fault'])
fit_time = time.time() - t0

t0 = time.time()
val_prob = starter_model.predict_proba(val)[:, 1]
predict_time = time.time() - t0

THRESHOLD = 0.5
val_verdict = (val_prob >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(val['is_true_fault'], val_verdict).ravel()
val_f1 = f1_score(val['is_true_fault'], val_verdict)
val_prec = precision_score(val['is_true_fault'], val_verdict, zero_division=0)
val_rec = recall_score(val['is_true_fault'], val_verdict, zero_division=0)

print(f"Starter Model: LogisticRegression on {FEATURES}")
print(f"Fit Time:     {fit_time:.4f}s")
print(f"Predict Time: {predict_time:.4f}s")
print(f"Threshold:    {THRESHOLD}")
print(f"True Negatives (tn):  {tn:>5}")
print(f"False Positives (fp): {fp:>5}  <- false alarms")
print(f"False Negatives (fn): {fn:>5}  <- missed faults")
print(f"True Positives (tp):  {tp:>5}  <- caught faults")
print(f"Precision:            {val_prec:.4f}")
print(f"Recall:               {val_rec:.4f}")
print(f"Validation F1:        {val_f1:.4f}")

# Generate starter submission.csv
test_verdict = (starter_model.predict_proba(test)[:, 1] >= THRESHOLD).astype(int)
sub = pd.DataFrame({'log_id': test['log_id'], 'is_true_fault': test_verdict})
sub.to_csv('submission_starter.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("\nGenerated submission.csv and saved copy submission_starter.csv")

print("\nRunning check_submission.py:")
valid = check("submission.csv")
print(f"Submission check result: {valid}")
