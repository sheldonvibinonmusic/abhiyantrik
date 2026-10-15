import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

# Test TF-IDF alone on message
tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=1000)
X_train_tfidf = tfidf.fit_transform(train['message'].fillna(''))
X_val_tfidf = tfidf.transform(val['message'].fillna(''))

clf = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
clf.fit(X_train_tfidf, train['is_true_fault'])

val_prob = clf.predict_proba(X_val_tfidf)[:, 1]

for th in [0.4, 0.45, 0.5, 0.55, 0.6]:
    pred = (val_prob >= th).astype(int)
    print(f"TF-IDF LogisticRegression (th={th}): F1 = {f1_score(val['is_true_fault'], pred):.4f}, Prec={precision_score(val['is_true_fault'], pred):.4f}, Rec={recall_score(val['is_true_fault'], pred):.4f}")

# Look at most positive and negative coefficients in TF-IDF
feature_names = np.array(tfidf.get_feature_names_out())
coefs = clf.coef_[0]
top_pos_idx = np.argsort(coefs)[-20:]
top_neg_idx = np.argsort(coefs)[:20]

print("\nTop 20 words predicting TRUE FAULT (positive):")
for idx in reversed(top_pos_idx):
    print(f"  {feature_names[idx]:<25} {coefs[idx]:.3f}")

print("\nTop 20 words predicting NOISE/LIE (negative):")
for idx in top_neg_idx:
    print(f"  {feature_names[idx]:<25} {coefs[idx]:.3f}")
