import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'
test['is_true_fault'] = np.nan

all_logs = pd.concat([train, val, test], ignore_index=True)
all_logs['ts'] = pd.to_datetime(all_logs['timestamp'])
all_logs = all_logs.sort_values('ts').reset_index(drop=True)

print("First 20 logs in chronological order:")
print(all_logs[['log_id', 'split', 'timestamp', 'module', 'mode', 'node', 'status', 'is_true_fault']].head(20))

print("\nDistribution of splits over time:")
all_logs['day'] = all_logs['ts'].dt.date
print(all_logs.groupby(['day', 'split']).size().unstack(fill_value=0))
