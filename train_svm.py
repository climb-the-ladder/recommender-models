#!/usr/bin/env python3
# train_svm.py

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
from sklearn.svm import SVC

# ───────────────────── Settings ─────────────────────
SEED        = 42
TEST_SIZE   = 0.2
DATA_PATH   = "processed_dataset.csv"
OUTPUT_DIR  = "career_svm"

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
svc = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=SEED
)
pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('smote', smote),
    ('clf', svc)
])

param_grid = {
    'clf__C': [1, 10, 50],
    'clf__gamma': ['scale', 'auto']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=6,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    random_state=SEED
)
search.fit(X_tr, y_tr)
best_svm = search.best_estimator_

# ─────────────────── Evaluation ───────────────────
y_pred = best_svm.predict(X_te)
print("SVM Test Accuracy :", accuracy_score(y_te, y_pred))
print("SVM Macro‑F1     :", f1_score(y_te, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=classes))

# ─────────────────── Save Artifacts ───────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_svm,           f"{OUTPUT_DIR}/svm_pipeline.pkl")
joblib.dump(le,                 f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\n✔️ Artifacts saved in {OUTPUT_DIR}/")
