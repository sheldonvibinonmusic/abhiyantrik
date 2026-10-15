import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

for mod in train['module'].unique():
    print("=" * 60)
    print(f"MODULE: {mod}")
    print("=" * 60)
    sub = train[train['module'] == mod]
    f = sub[sub['is_true_fault'] == 1]['reading'].dropna()
    nf = sub[sub['is_true_fault'] == 0]['reading'].dropna()
    print(f"Fault (1) count={len(f)}: min={f.min():.2f}, 1%={f.quantile(0.01):.2f}, 5%={f.quantile(0.05):.2f}, 50%={f.median():.2f}, 95%={f.quantile(0.95):.2f}, 99%={f.quantile(0.99):.2f}, max={f.max():.2f}")
    print(f"Non-Fault (0) count={len(nf)}: min={nf.min():.2f}, 1%={nf.quantile(0.01):.2f}, 5%={nf.quantile(0.05):.2f}, 50%={nf.median():.2f}, 95%={nf.quantile(0.95):.2f}, 99%={nf.quantile(0.99):.2f}, max={nf.max():.2f}")
    
    # Check if there are thresholds where fault probability spikes
    bins = pd.qcut(sub['reading'].dropna(), q=10, duplicates='drop')
    print("\nReading deciles vs fault rate:")
    print(sub.groupby(bins, observed=False)['is_true_fault'].agg(['count', 'mean']))

print("\n" + "=" * 60)
print("CHECKING FOR SPECIAL CONDITIONS")
print("=" * 60)
# Look at message vs fault:
print("Top 10 highest fault rate messages:")
msg_stat = train.groupby('message')['is_true_fault'].agg(['count', 'mean'])
print(msg_stat[msg_stat['count'] >= 5].sort_values('mean', ascending=False).head(15))

print("\nTop 10 lowest fault rate messages:")
print(msg_stat[msg_stat['count'] >= 5].sort_values('mean', ascending=True).head(15))
