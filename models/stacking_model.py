#!/usr/bin/env python3
# train_stacking.py

import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.lgbm_model import build_lgbm
from src.models.rf_model import build_rf
from src.models.svm_model import build_svm
from src.models.mlp_model import build_mlp

# ─────────────────── Settings ───────────────────
SEED        = 42
TEST_SIZE   = 0.2
DATA_PATH   = "processed_dataset.csv"
OUTPUT_DIR  = "career_stacking"

# ────────── Load & Encode Data ──────────
df      = pd.read_csv(DATA_PATH)
X       = df.drop(columns=["career_aspiration"])
y       = df["career_aspiration"]

le      = LabelEncoder().fit(y)
y_enc   = le.transform(y)
classes = le.classes_

# ────────── Train/Test Split ──────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,
    random_state=SEED
)

# ────────── Preprocessing & SMOTE Pipeline ──────────
numeric_features = X.columns.tolist()
preprocessor     = ColumnTransformer(
    [('scale', StandardScaler(), numeric_features)]
)
smote            = SMOTE(random_state=SEED)

# ────────── Build & Fit Base Estimators ──────────
base_estimators = {
    'lgbm': build_lgbm(len(classes)),
    'rf':   build_rf(),
    'svm':  build_svm(),
    'mlp':  build_mlp()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fitted_estimators = {}
for name, clf in base_estimators.items():
    pipe = ImbPipeline([
        ('pre',  preprocessor),
        ('smote', smote),
        ('clf',   clf)
    ])
    pipe.fit(X_tr, y_tr)
    fitted_estimators[name] = pipe

# ────────── Build & Train Stacking Model ──────────
stack = StackingClassifier(
    estimators=[(n, est) for n, est in fitted_estimators.items()],
    final_estimator=LogisticRegression(max_iter=500),
    cv=cv,
    n_jobs=-1
)
stack.fit(X_tr, y_tr)

# ────────── Evaluation ──────────
y_pred = stack.predict(X_te)
print("Stacking Test Accuracy :", accuracy_score(y_te, y_pred))
print("Stacking Macro‑F1     :", f1_score(y_te, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=classes))

# ────────── Save Artifacts ──────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(stack,                 f"{OUTPUT_DIR}/stacking_model.pkl")
joblib.dump(le,                    f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\n✔️ Artifacts saved in {OUTPUT_DIR}/")
