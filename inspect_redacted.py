import pandas as pd

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

print("=== CHECKING 'redacted' IN TRAIN ===")
r_train = train[train['message'].str.contains('redacted', case=False, na=False)]
print(f"Count in train: {len(r_train)}")
print("Fault rate when 'redacted' is in message:", r_train['is_true_fault'].mean())
print(r_train[['status', 'module', 'reading', 'message', 'is_true_fault']].head(10))

print("\n=== CHECKING 'redacted' IN VAL ===")
r_val = val[val['message'].str.contains('redacted', case=False, na=False)]
print(f"Count in val: {len(r_val)}")
print("Fault rate in val:", r_val['is_true_fault'].mean())

print("\n=== CHECKING 'redacted' IN TEST ===")
r_test = test[test['message'].str.contains('redacted', case=False, na=False)]
print(f"Count in test: {len(r_test)}")

print("\nWhat other strange tokens appear in messages?")
for t in ['fault', 'fail', 'corrupt', 'unknown', 'override', 'bypass', 'rogue', 'lie', 'ghost', 'truth', 'emergency', 'manual']:
    c_tr = train['message'].str.contains(t, case=False, na=False).sum()
    if c_tr > 0:
        f_tr = train[train['message'].str.contains(t, case=False, na=False)]['is_true_fault'].mean()
        print(f"Token '{t}': train count={c_tr}, fault_rate={f_tr:.3f}")
