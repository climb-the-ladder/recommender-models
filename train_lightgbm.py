"""
train_lightgbm.py
=================
Train a multiclass LightGBM model on processed_dataset.csv
and save model + label encoder in recommender‑models/artifacts/.

Run:
    python -m pip install -r requirements.txt   # see deps below
    python recommender-models/train_lightgbm.py
"""

import os, math, warnings, joblib, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import lightgbm as lgb

# ─────────────────────────── Settings ───────────────────────────
SEED        = 42
TEST_SIZE   = 0.20         # 20 % hold‑out
USE_SMOTE   = True         # toggle balancing strategy
N_ESTIMATORS= 800

# ─────────────────────────── Dataset ────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH   = os.path.join(SCRIPT_DIR, "processed_dataset.csv")

df = pd.read_csv(CSV_PATH)
X  = df.drop(columns=["career_aspiration"])
y  = df["career_aspiration"]

print(f"Loaded {CSV_PATH}  →  {df.shape[0]} rows / {df.shape[1]} cols")

# ───────────────────── Encode & balance targets ─────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

if USE_SMOTE:
    sm  = SMOTE(random_state=SEED)
    X_b, y_b = sm.fit_resample(X, y_enc)
    class_weight = None
    print(f"After SMOTE: {X_b.shape[0]} rows (balanced)")
else:
    freq = Counter(y_enc)
    class_weight = {cls: len(y_enc) / (len(freq) * cnt)
                    for cls, cnt in freq.items()}
    X_b, y_b = X, y_enc
    print("Using inverse‑frequency class weights")

# ───────────────────────── Train / test split ──────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_b, y_b, test_size=TEST_SIZE, stratify=y_b, random_state=SEED
)

# ─────────────────────────── Train model ────────────────────────
lgbm = LGBMClassifier(
    objective      ="multiclass",
    num_class      = len(le.classes_),
    n_estimators   = N_ESTIMATORS,
    learning_rate  = 0.03,
    num_leaves     = 128,
    subsample      = 0.8,
    colsample_bytree=0.8,
    reg_lambda     = 1.0,
    class_weight   = class_weight,
    random_state   = SEED,
)

lgbm.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# ─────────────────────────── Evaluate ───────────────────────────
y_pred = lgbm.predict(X_te)
print("\nClassification report:")
print(classification_report(y_te, y_pred, target_names=le.classes_))

print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
print(f"Macro‑F1 : {f1_score(y_te, y_pred, average='macro'):.4f}")

# ─────────────────────────── Persist ────────────────────────────
ART_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)
joblib.dump(lgbm, os.path.join(ART_DIR, "career_lgbm.pkl"))
joblib.dump(le,    os.path.join(ART_DIR, "label_encoder.pkl"))
print(f"\nSaved artifacts to {ART_DIR}")
