import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')

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

print("=== COARSE FEATURE COMBINATIONS ===")
grp = train.groupby(['module', 'status', 'prefix', 'payload'])['is_true_fault'].agg(['count', 'mean'])
print(f"Total (module, status, prefix, payload) combinations: {len(grp)}")
print("Combinations with mean == 1.0 (pure fault):", (grp['mean'] == 1.0).sum())
print("Combinations with mean == 0.0 (pure non-fault):", (grp['mean'] == 0.0).sum())
print("Combinations with 0.0 < mean < 1.0 (mixed):", ((grp['mean'] > 0.0) & (grp['mean'] < 1.0)).sum())

# What is the entropy or Bayes error if we only use module, status, prefix, payload?
p = grp['mean']
n = grp['count']
# Expected log loss or error rate of majority class within group
majority_acc = (n * np.maximum(p, 1 - p)).sum() / n.sum()
print(f"Majority-class theoretical accuracy on (module, status, prefix, payload): {majority_acc:.4f}")
