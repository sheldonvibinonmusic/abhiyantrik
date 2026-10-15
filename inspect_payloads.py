import pandas as pd
import numpy as np
import re

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

print("1. Missing reading vs fault:")
print(train[train['reading'].isna()]['is_true_fault'].value_counts(normalize=True))
print(f"Total missing in train: {train['reading'].isna().sum()}, fault rate: {train[train['reading'].isna()]['is_true_fault'].mean():.4f}")

# Extract bracket content: [module reading_in_msg unit]
def extract_bracket(msg):
    match = re.search(r'\[(.*?)\]', msg)
    if match:
        parts = match.group(1).split()
        if len(parts) >= 3:
            return parts[0], float(parts[1]), parts[2]
        elif len(parts) == 2:
            try:
                return parts[0], float(parts[1]), ""
            except:
                return parts[0], np.nan, parts[1]
    return None, np.nan, None

bracket_info = train['message'].apply(extract_bracket)
train['msg_module'] = [b[0] for b in bracket_info]
train['msg_reading'] = [b[1] for b in bracket_info]
train['msg_unit'] = [b[2] for b in bracket_info]

print("\n2. Comparing dataframe reading vs message reading:")
diff = (train['reading'] - train['msg_reading']).dropna()
print(f"Max abs diff between df reading and msg reading: {diff.abs().max()}")
print(f"Are msg_module and module column identical? {(train['module'] == train['msg_module']).all()}")

# Look at the payload text between prefix and bracket
def extract_payload(msg):
    # remove prefix "NINE: something."
    # remove bracket "[...]"
    m = re.sub(r'\[.*?\]', '', msg).strip()
    if 'NINE:' in m:
        m = m.replace('NINE:', '').strip()
    return m

train['payload'] = train['message'].apply(extract_payload)
print("\n3. Unique payloads and their fault rates (sorted by fault rate descending):")
p_stats = train.groupby('payload')['is_true_fault'].agg(['count', 'mean']).sort_values(['mean', 'count'], ascending=[False, False])
print(p_stats.to_string())

print("\n4. Status + Module combination fault rates:")
sm = train.groupby(['module', 'status'])['is_true_fault'].agg(['count', 'mean'])
print(sm)

print("\n5. Correlation between numerical features and fault:")
print(train[['reading', 'delay_ms', 'retry_count', 'is_true_fault']].corr())
