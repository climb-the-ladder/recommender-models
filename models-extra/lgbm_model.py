#!/usr/bin/env python3
# train_lgbm.py
#angel-version of lgbm model

import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb

# ───────────────────── Settings ─────────────────────
SEED        = 42
TEST_SIZE   = 0.2
DATA_PATH   = "processed_dataset.csv"
OUTPUT_DIR  = "career_lgbm"

# ────────────────── Load & Encode Data ──────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["career_aspiration"])
y  = df["career_aspiration"]

le     = LabelEncoder().fit(y)
y_enc  = le.transform(y)
classes = le.classes_

# ────────────────── Train/Test Split ──────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,
    random_state=SEED
)

# ────────── Preprocessing & SMOTE Pipeline ──────────
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    [('scale', StandardScaler(), numeric_features)]
)
smote = SMOTE(random_state=SEED)

# ────────── Model & Hyperparameter Search ──────────
lgbm_clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(classes),
    random_state=SEED,
    verbosity=-1
)
pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('smote', smote),
    ('clf', lgbm_clf)
])

param_grid = {
    'clf__n_estimators': [500, 1000, 1500],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__num_leaves': [31, 63, 127]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    random_state=SEED
)
search.fit(X_tr, y_tr)
best_lgbm = search.best_estimator_

# ─────────────────── Evaluation ───────────────────
y_pred = best_lgbm.predict(X_te)
print("LightGBM Test Accuracy :", accuracy_score(y_te, y_pred))
print("LightGBM Macro‑F1     :", f1_score(y_te, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=classes))

# ─────────────────── Save Artifacts ───────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_lgbm,           f"{OUTPUT_DIR}/lgbm_pipeline.pkl")
joblib.dump(le,                  f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\n✔️ Artifacts saved in {OUTPUT_DIR}/")
