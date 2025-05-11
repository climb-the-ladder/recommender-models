#!/usr/bin/env python3
# train_mlp.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────── Settings ───────────────────
SEED        = 42
LR          = 1e-3
EPOCHS      = 100
BATCH_SIZE  = 64
TEST_SIZE   = 0.2
DATA_PATH   = "processed_dataset.csv"
OUTPUT_DIR  = "career_mlp"

# ────────── Reproducibility ──────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ────────── Load & Encode Data ──────────
df       = pd.read_csv(DATA_PATH)
X        = df.drop(columns=["career_aspiration"]).values.astype(np.float32)
y        = df["career_aspiration"].values

scaler       = StandardScaler()
X_scaled     = scaler.fit_transform(X)

le           = LabelEncoder().fit(y)
y_enc        = le.transform(y)
classes      = le.classes_

# ────────── Train/Test Split ──────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,
    random_state=SEED
)

# ────────── Apply SMOTE ──────────
smote       = SMOTE(random_state=SEED)
X_tr, y_tr  = smote.fit_resample(X_tr, y_tr)

# ────────── Tensor Conversion ──────────
X_tr_tensor = torch.tensor(X_tr)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
X_te_tensor = torch.tensor(X_te)
y_te_tensor = torch.tensor(y_te, dtype=torch.long)

train_ds     = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ────────── Define MLP ──────────
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleMLP(X_tr_tensor.shape[1], len(classes)).to(device)

# ────────── Loss & Optimizer ──────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ────────── Training Loop ──────────
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} – Loss: {epoch_loss / len(train_ds):.4f}")

# ────────── Evaluation ──────────
model.eval()
with torch.no_grad():
    logits = model(X_te_tensor.to(device))
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_te_tensor.numpy()

print("\nMLP Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
print("MLP Test Accuracy :", accuracy_score(y_true, y_pred))
print("MLP Macro‑F1     :", f1_score(y_true, y_pred, average="macro"))

# ────────── Save Artifacts ──────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(model.state_dict(),            f"{OUTPUT_DIR}/model.pth")
joblib.dump(le,                           f"{OUTPUT_DIR}/label_encoder.pkl")
joblib.dump(scaler,                       f"{OUTPUT_DIR}/scaler.pkl")
print(f"\n✔️ Artifacts saved in {OUTPUT_DIR}/")
