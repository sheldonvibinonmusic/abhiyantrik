import json

with open('starter_round1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 9: Act 2 Model definition
act2_code = """from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import numpy as np
import re

NOMINAL_CENTERS = {
    'Lab-Power': 12.18,
    'Life-Support': 21.11,
    'Antenna': 43.13,
    'Notice-Bus': 313.09
}
NOMINAL_SCALES = {
    'Lab-Power': 1.05,
    'Life-Support': 1.67,
    'Antenna': 7.95,
    'Notice-Bus': 56.31
}

class LogFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        
        def parse_msg(m):
            m = str(m) if pd.notna(m) else ""
            bracket = re.search(r'\\[(.*?)\\]', m)
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

        prefixes, payloads, brackets = zip(*df['message'].apply(parse_msg))
        out['prefix'] = pd.Series(prefixes, index=df.index, dtype='category')
        out['payload'] = pd.Series(payloads, index=df.index, dtype='category')
        out['is_redacted'] = [1 if 'REDACTED' in b else 0 for b in brackets]
        out['reading_isna'] = df['reading'].isna().astype(int)
        out['reading'] = df['reading']
        
        z_devs = []
        abs_z_devs = []
        sq_z_devs = []
        is_extreme = []
        for mod, r in zip(df['module'], df['reading']):
            if pd.isna(r) or mod not in NOMINAL_CENTERS:
                z_devs.append(0.0)
                abs_z_devs.append(0.0)
                sq_z_devs.append(0.0)
                is_extreme.append(0)
            else:
                z = (r - NOMINAL_CENTERS[mod]) / NOMINAL_SCALES[mod]
                z_devs.append(z)
                abs_z_devs.append(abs(z))
                sq_z_devs.append(z ** 2)
                is_extreme.append(1 if abs(z) > 1.5 else 0)
                
        out['z_dev'] = z_devs
        out['abs_z_dev'] = abs_z_devs
        out['sq_z_dev'] = sq_z_devs
        out['is_extreme_reading'] = is_extreme
        out['delay_ms'] = df['delay_ms']
        out['delay_log'] = np.log1p(df['delay_ms'].clip(lower=0))
        out['delay_high'] = (df['delay_ms'] > 450).astype(int)
        out['retry_count'] = df['retry_count']
        out['has_retries'] = (df['retry_count'] > 0).astype(int)
        out['module'] = df['module'].astype('category')
        out['mode'] = df['mode'].astype('category')
        out['node'] = df['node'].astype('category')
        out['status'] = df['status'].astype('category')
        
        calm = out['prefix'].isin(['all is well', 'crew need not worry', 'nothing to see'])
        severe = out['status'].isin(['CRIT', 'ERROR'])
        out['calm_yet_severe'] = (calm & severe).astype(int)
        
        benign_payloads = [
            'scheduled calibration', 'self-test pattern', 'duplicate of earlier alert',
            'simulated drill event', 'recovered after retry', 'sensor warming up'
        ]
        out['is_benign_routine'] = out['payload'].isin(benign_payloads).astype(int)
        out['benign_yet_severe'] = (out['is_benign_routine'] & severe).astype(int)
        
        failure_payloads = [
            'dish actuator stalled', 'carrier drift beyond tolerance', 'phase lock lost',
            'breaker tripped twice', 'overvolt on rail B', 'cascade brownout detected',
            'O2 partial pressure falling', 'seal breach suspected', 'scrubber saturation',
            'bus arbitration deadlock', 'queue overflow, packets dropped', 'heartbeat missing from 3 nodes'
        ]
        out['is_failure_payload'] = out['payload'].isin(failure_payloads).astype(int)
        
        return out

class EnsembleRecalibrationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else ['prefix', 'payload', 'module', 'mode', 'node', 'status']
        self.clf1 = HistGradientBoostingClassifier(
            categorical_features=self.cat_cols,
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=25,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        )
        self.clf2 = HistGradientBoostingClassifier(
            categorical_features=self.cat_cols,
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=25,
            min_samples_leaf=20,
            class_weight='balanced',
            l2_regularization=2.0,
            random_state=123
        )
        
    def fit(self, X, y):
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self
        
    def predict_proba(self, X):
        p1 = self.clf1.predict_proba(X)
        p2 = self.clf2.predict_proba(X)
        return 0.60 * p1 + 0.40 * p2
        
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.33).astype(int)

model = Pipeline([
    ('features', LogFeatureExtractor()),
    ('classifier', EnsembleRecalibrationClassifier())
])

progress("Training Recalibration Protocol v3")
model.fit(train, train['is_true_fault'])
sysmsg("Recalibration Protocol v3 trained successfully.")
"""

nb['cells'][9]['source'] = [line + '\n' for line in act2_code.strip().split('\n')]
nb['cells'][9]['outputs'] = []

# Cell 11: Act 3 Threshold
act3_code = """val_prob = model.predict_proba(val)[:, 1]
THRESHOLD = 0.33  # Calibrated on out-of-fold and validation F1
"""
nb['cells'][11]['source'] = [line + '\n' for line in act3_code.strip().split('\n')]
nb['cells'][11]['outputs'] = []

# Clear outputs of remaining cells so there are no old error traces
for idx in [14, 16, 17, 19]:
    nb['cells'][idx]['outputs'] = []

with open('starter_round1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated starter_round1.ipynb successfully.")
