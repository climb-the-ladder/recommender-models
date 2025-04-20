#!/usr/bin/env python3
"""
train_fm_pytorch.py

Multi‑class Factorization Machine implemented in PyTorch,
trained on processed_dataset.csv with cross‑entropy loss.

Usage:
    # 1. Create & activate a virtualenv, then:
    pip install torch pandas scikit-learn joblib
    python train_fm_pytorch.py

Outputs:
    career_fm_pytorch.pkl    # the trained PyTorch model
    label_encoder.pkl        # sklearn LabelEncoder
    scaler.pkl               # sklearn StandardScaler
"""

import os
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ─────────────────── Hyperparameters ───────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR           = 1e-2
EPOCHS       = 100
BATCH_SIZE   = 64
RANK         = 8
TEST_SIZE    = 0.2
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────── Dataset Prep ──────────────────────────
df = pd.read_csv("processed_dataset.csv")
X = df.drop(columns=["career_aspiration"]).values.astype(np.float32)
y = df["career_aspiration"].values

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# encode labels
le    = LabelEncoder()
y_enc = le.fit_transform(y)

# train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,
    random_state=SEED,
)

# convert to tensors
X_tr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
X_te = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
y_tr = torch.tensor(y_tr, dtype=torch.long,    device=DEVICE)
y_te = torch.tensor(y_te, dtype=torch.long,    device=DEVICE)

N_FEATURES = X_tr.shape[1]
N_CLASSES  = len(le.classes_)

# ───────────────── Multi‑class FM Model ─────────────────────────
class FactorizationMachine(nn.Module):
    def __init__(self, n_features, n_classes, k):
        super().__init__()
        # linear term
        self.linear = nn.Linear(n_features, n_classes, bias=True)
        # factor matrix: shape (n_features, k, n_classes)
        self.V = nn.Parameter(
            torch.randn(n_features, k, n_classes) * 0.01
        )

    def forward(self, x):
        # x: [batch, n_features]
        # linear output
        lin_out = self.linear(x)  # [batch, n_classes]

        # second‑order interactions per class
        # we'll do a small loop over classes (n_classes ~ 15)
        batch_size = x.size(0)
        interactions = []
        # x_i * V_i,f,c summed over i for each factor f and class c:
        # per class:
        for c in range(N_CLASSES):
            V_c = self.V[:, :, c]          # [n_features, k]
            x_v = x @ V_c                  # [batch, k]  sum_i x_i * V_i,f,c
            x2_v2 = (x * x) @ (V_c * V_c)  # [batch, k]
            # sum over k: 0.5 * ( (sum_i x_i v_i)^2 – sum_i x_i^2 v_i^2 )
            inter_c = 0.5 * (x_v * x_v - x2_v2).sum(dim=1, keepdim=True)
            interactions.append(inter_c)
        inter_out = torch.cat(interactions, dim=1)  # [batch, n_classes]

        return lin_out + inter_out

model = FactorizationMachine(N_FEATURES, N_CLASSES, RANK).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─────────────────────── Training Loop ─────────────────────────
model.train()
dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if epoch % 10 == 0 or epoch == 1:
        avg = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:3d}/{EPOCHS} – loss: {avg:.4f}")

# ─────────────────────── Evaluation ────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(X_te)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_te.cpu().numpy()

print("\n📊 Classification report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Macro‑F1 :", f1_score(y_true, y_pred, average="macro"))

# ─────────────────────── Persist Model ─────────────────────────
os.makedirs("career_fm_pytorch", exist_ok=True)
torch.save(model.state_dict(), "career_fm_pytorch/fm_state_dict.pth")
joblib.dump(le,     "career_fm_pytorch/label_encoder.pkl")
joblib.dump(scaler, "career_fm_pytorch/scaler.pkl")
print("\n💾 Saved FM PyTorch model + encoder + scaler to career_fm_pytorch/")
