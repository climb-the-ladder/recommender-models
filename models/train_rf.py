#!/usr/bin/env python3
# train_rf.py

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
from sklearn.ensemble import RandomForestClassifier

# ─────────────────── Settings ───────────────────
SEED        = 42
TEST_SIZE   = 0.2
DATA_PATH   = "processed_dataset.csv"
OUTPUT_DIR  = "career_rf"

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

# ────────── Model & Hyperparameter Search ──────────
rf_clf = RandomForestClassifier(
    random_state=SEED,
    class_weight='balanced'
)
pipeline = ImbPipeline([
    ('pre',  preprocessor),
    ('smote', smote),
    ('clf',   rf_clf)
])

param_grid = {
    'clf__n_estimators': [100, 300, 500],
    'clf__max_depth': [None, 10, 20],
    'clf__max_features': ['sqrt', 'log2']
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
best_rf = search.best_estimator_

# ────────── Evaluation ──────────
y_pred = best_rf.predict(X_te)
print("RandomForest Test Accuracy :", accuracy_score(y_te, y_pred))
print("RandomForest Macro‑F1     :", f1_score(y_te, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=classes))

# ────────── Save Artifacts ──────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_rf,            f"{OUTPUT_DIR}/rf_pipeline.pkl")
joblib.dump(le,                 f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\n✔️ Artifacts saved in {OUTPUT_DIR}/")
