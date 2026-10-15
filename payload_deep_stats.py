import pandas as pd
import re

train = pd.read_csv('train.csv')

def parse_msg(m):
    bracket = re.search(r'\[(.*?)\]', m)
    bracket_content = bracket.group(1) if bracket else ""
    text_before = m[:bracket.start()].strip() if bracket else m.strip()
    if text_before.startswith('NINE:'):
        text_before = text_before[5:].strip()
    parts = text_before.split('. ', 1)
    if len(parts) == 2:
        prefix, payload = parts[0].strip(), parts[1].strip()
    else:
        prefix, payload = "", parts[0].strip()
    return prefix, payload, bracket_content

train['prefix'], train['payload'], train['bracket'] = zip(*train['message'].apply(parse_msg))

payload_stats = train.groupby('payload')['is_true_fault'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print("=== PAYLOAD FAULT RATES ===")
print(payload_stats)

prefix_stats = train.groupby('prefix')['is_true_fault'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print("\n=== PREFIX FAULT RATES ===")
print(prefix_stats)

print("\n=== PREFIX x STATUS FAULT RATES ===")
ps = train.groupby(['prefix', 'status'])['is_true_fault'].agg(['count', 'mean'])
print(ps)
