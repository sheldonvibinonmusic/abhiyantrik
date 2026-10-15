import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

print("=== CHECKING FOR IDENTICAL / CONFLICTING RECORDS IN TRAIN ===")
feature_cols = ['module', 'mode', 'node', 'status', 'reading', 'delay_ms', 'retry_count', 'message']
dup_all = train.duplicated(subset=feature_cols, keep=False)
print(f"Exact feature duplicates (including delay & reading): {dup_all.sum()} rows")

# Exclude continuous noise: module, status, message
dup_msg = train.duplicated(subset=['module', 'status', 'message'], keep=False)
print(f"Rows sharing (module, status, message): {dup_msg.sum()} rows")

# Check label entropy / consistency for identical (module, status, message)
grouped = train.groupby(['module', 'status', 'message'])['is_true_fault'].agg(['count', 'mean'])
mixed = grouped[(grouped['count'] > 1) & (grouped['mean'] > 0) & (grouped['mean'] < 1)]
pure_1 = grouped[(grouped['count'] > 1) & (grouped['mean'] == 1)]
pure_0 = grouped[(grouped['count'] > 1) & (grouped['mean'] == 0)]

print(f"Total unique (module, status, message) combinations: {len(grouped)}")
print(f"Combinations with >1 occurrence: {(grouped['count'] > 1).sum()}")
print(f"Pure non-fault combinations (100% 0): {len(pure_0)} (covers {pure_0['count'].sum()} rows)")
print(f"Pure true fault combinations (100% 1): {len(pure_1)} (covers {pure_1['count'].sum()} rows)")
print(f"Mixed combinations (some 0, some 1): {len(mixed)} (covers {mixed['count'].sum()} rows)")

print("\nSample mixed combinations:")
print(mixed.head(10))

print("\n=== SENSOR READING DEVIATIONS VS TARGET IN MIXED GROUPS ===")
# Is the variation within mixed groups explained by the sensor reading or delay?
print("Let's check if reading within a mixed group cleanly separates fault vs non-fault:")
for idx, (combo, row) in enumerate(mixed.head(3).iterrows()):
    m, s, msg = combo
    subset = train[(train['module'] == m) & (train['status'] == s) & (train['message'] == msg)]
    print(f"\nCombo {idx+1}: {combo}")
    print(subset[['reading', 'delay_ms', 'retry_count', 'is_true_fault']])
