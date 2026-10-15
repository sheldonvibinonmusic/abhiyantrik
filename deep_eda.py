import pandas as pd
import numpy as np
from collections import Counter
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

print("=" * 60)
print("A. STATUS vs IS_TRUE_FAULT")
print("=" * 60)
ct_status = pd.crosstab(train['status'], train['is_true_fault'], normalize='index')
print(ct_status)
print("\nCounts:")
print(pd.crosstab(train['status'], train['is_true_fault'], margins=True))

print("\n" + "=" * 60)
print("B. MODULE vs IS_TRUE_FAULT")
print("=" * 60)
ct_mod = pd.crosstab(train['module'], train['is_true_fault'], normalize='index')
print(ct_mod)

print("\n" + "=" * 60)
print("C. MODE vs IS_TRUE_FAULT")
print("=" * 60)
ct_mode = pd.crosstab(train['mode'], train['is_true_fault'], normalize='index')
print(ct_mode)

print("\n" + "=" * 60)
print("D. NODE vs IS_TRUE_FAULT")
print("=" * 60)
ct_node = pd.crosstab(train['node'], train['is_true_fault'], normalize='index')
print(ct_node)

print("\n" + "=" * 60)
print("E. PER-MODULE READING STATS FOR FAULT vs NON-FAULT")
print("=" * 60)
for m in train['module'].unique():
    print(f"\n--- Module: {m} ---")
    sub_fault = train[(train['module'] == m) & (train['is_true_fault'] == 1)]['reading']
    sub_non = train[(train['module'] == m) & (train['is_true_fault'] == 0)]['reading']
    print(f"Fault (1) reading: count={len(sub_fault.dropna())}, mean={sub_fault.mean():.2f}, std={sub_fault.std():.2f}, min={sub_fault.min():.2f}, 50%={sub_fault.median():.2f}, max={sub_fault.max():.2f}")
    print(f"Non-fault (0) reading: count={len(sub_non.dropna())}, mean={sub_non.mean():.2f}, std={sub_non.std():.2f}, min={sub_non.min():.2f}, 50%={sub_non.median():.2f}, max={sub_non.max():.2f}")

print("\n" + "=" * 60)
print("F. DELAY_MS & RETRY_COUNT BY FAULT")
print("=" * 60)
print("delay_ms:")
print(train.groupby('is_true_fault')['delay_ms'].describe())
print("\nretry_count:")
print(pd.crosstab(train['retry_count'], train['is_true_fault'], normalize='index'))

print("\n" + "=" * 60)
print("G. MESSAGE TEXT PATTERNS")
print("=" * 60)
# Extract prefix before first period or colon
prefixes = train['message'].apply(lambda x: x.split('.')[0] if '.' in x else x.split(':')[0])
print("Top message prefixes by fault rate:")
p_df = pd.DataFrame({'prefix': prefixes, 'target': train['is_true_fault']})
print(p_df.groupby('prefix')['target'].agg(['count', 'mean']).sort_values('count', ascending=False).head(15))

# Extract the message text without the trailing [Module reading unit]
def clean_msg(msg):
    # remove trailing [Module ...]
    m = re.sub(r'\[.*?\]', '', msg).strip()
    return m

train['clean_msg'] = train['message'].apply(clean_msg)
print("\nTop clean messages by fault rate:")
cm_df = pd.DataFrame({'msg': train['clean_msg'], 'target': train['is_true_fault']})
print(cm_df.groupby('msg')['target'].agg(['count', 'mean']).sort_values('count', ascending=False).head(20))

# Word frequency differences
words_1 = Counter()
words_0 = Counter()
for msg in train[train['is_true_fault'] == 1]['clean_msg']:
    words_1.update(re.findall(r'\b\w+\b', msg.lower()))
for msg in train[train['is_true_fault'] == 0]['clean_msg']:
    words_0.update(re.findall(r'\b\w+\b', msg.lower()))

print("\nTop distinctive words for faults (is_true_fault == 1):")
for w, c in words_1.most_common(25):
    rate = c / (c + words_0.get(w, 0))
    print(f"  {w:<15} fault_count={c:<5} non_fault_count={words_0.get(w,0):<5} fault_ratio={rate:.3f}")
