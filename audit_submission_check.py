import pandas as pd
from check_submission import check

sub = pd.read_csv('submission.csv')
test = pd.read_csv('test.csv')

print("=== LIGHTWEIGHT SUBMISSION AUDIT ===")
print("File shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Header check:", list(sub.columns) == ['log_id', 'is_true_fault'])
print("Total rows:", len(sub))
print("Unique IDs count:", sub['log_id'].nunique())
print("Null count in log_id:", sub['log_id'].isnull().sum())
print("Null count in is_true_fault:", sub['is_true_fault'].isnull().sum())
print("Unique values in is_true_fault:", sorted(sub['is_true_fault'].unique()))
print("Fault count (1):", int((sub['is_true_fault'] == 1).sum()))
print("Noise count (0):", int((sub['is_true_fault'] == 0).sum()))
print("First log_id:", sub['log_id'].iloc[0], "Last log_id:", sub['log_id'].iloc[-1])
print("IDs match test.csv exactly in order:", (sub['log_id'] == test['log_id']).all())

print("\n=== RUNNING OFFICIAL FORMAT CHECKER ===")
res = check('submission.csv')
print("Checker returned:", res)
