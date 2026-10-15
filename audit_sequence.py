import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
train['ts'] = pd.to_datetime(train['timestamp'])
train = train.sort_values('ts').reset_index(drop=True)

# Time delta to previous log
train['dt_prev_sec'] = train['ts'].diff().dt.total_seconds()
print("Time delta summary (seconds):")
print(train['dt_prev_sec'].describe())

# Auto-correlation of is_true_fault over chronological order
lags = [1, 2, 3, 5, 10]
for lag in lags:
    corr = train['is_true_fault'].autocorr(lag=lag)
    print(f"Lag {lag} autocorrelation: {corr:.4f}")

# Time delta between logs of the SAME module
for mod in train['module'].unique():
    mod_df = train[train['module'] == mod].copy()
    mod_df['dt'] = mod_df['ts'].diff().dt.total_seconds()
    corr = mod_df['is_true_fault'].autocorr(lag=1)
    print(f"Module {mod} Lag-1 autocorrelation: {corr:.4f}")
