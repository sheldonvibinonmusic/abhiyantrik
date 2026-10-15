import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')

# Load the experiment predictions or run OOF on Exp 4
# Let's inspect the OOF predictions of EXP 1 and EXP 4
# We can re-extract the errors from the out-of-fold predictions
from run_experiments_v2 import oof_probs_e1, best_oof_th_e1, oof_meta_preds, best_oof_th_e4, parse_msg

train['pred_e1'] = (oof_probs_e1 >= best_oof_th_e1).astype(int)
train['pred_e4'] = (oof_meta_preds >= best_oof_th_e4).astype(int)

# Deconstruct message syntax
prefixes, payloads, brackets = zip(*train['message'].apply(parse_msg))
train['prefix'] = prefixes
train['payload'] = payloads

# Classify error types for EXP_4
# TP: fault=1, pred=1
# TN: fault=0, pred=0
# FP: fault=0, pred=1 (crying wolf / fooled by NINE)
# FN: fault=1, pred=0 (missed fault / deceived by NINE)
conditions = [
    (train['is_true_fault'] == 1) & (train['pred_e4'] == 1),
    (train['is_true_fault'] == 0) & (train['pred_e4'] == 0),
    (train['is_true_fault'] == 0) & (train['pred_e4'] == 1),
    (train['is_true_fault'] == 1) & (train['pred_e4'] == 0)
]
choices = ['TP', 'TN', 'FP', 'FN']
train['error_type'] = np.select(conditions, choices, default='OTHER')

print("=== OVERALL OOF CONFUSION MATRIX ===")
print(train['error_type'].value_counts())

print("\n=== TOP PAYLOADS IN FALSE POSITIVES (FP) ===")
# Where does the model predict fault, but it's noise?
fp_df = train[train['error_type'] == 'FP']
print(fp_df.groupby(['module', 'status', 'payload']).size().sort_values(ascending=False).head(15))

print("\n=== TOP PAYLOADS IN FALSE NEGATIVES (FN) ===")
# Where does the model miss a real fault?
fn_df = train[train['error_type'] == 'FN']
print(fn_df.groupby(['module', 'status', 'prefix', 'payload']).size().sort_values(ascending=False).head(15))

print("\n=== STATUS BREAKDOWN BY ERROR TYPE ===")
print(pd.crosstab(train['status'], train['error_type'], normalize='index'))

print("\n=== PREFIX BREAKDOWN BY ERROR TYPE ===")
print(pd.crosstab(train['prefix'], train['error_type'], normalize='index'))

print("\n=== SENSOR READING DEVIATIONS IN FN vs TP ===")
print("True Positives sensor extreme rate:")
print(train[train['error_type'] == 'TP']['reading'].isna().mean())
print("False Negatives reading summary:")
print(train[train['error_type'] == 'FN']['reading'].describe())
