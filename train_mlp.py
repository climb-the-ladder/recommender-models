#!/usr/bin/env python3
# train_mlp.py

import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ───────────────────── Settings ─────────────────────
SEED       = 42
LR         = 1e-3
EPOCHS     = 100
BATCH_SIZE = 64
TEST_SIZE  = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)

# ────────────────── Load & prepare data ──────────────────
df = pd.read_csv("processed_dataset.csv")
X = df.drop(columns=["career_aspiration"]).values.astype(np.float32)
y = df["career_aspiration"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

le    = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

# compute class weights for loss
cls_wts = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_enc),
    y=y_enc
)
cls_wts = torch.tensor(cls_wts, dtype=torch.float32)

# train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,
    random_state=SEED
)

# to tensors
X_tr = torch.tensor(X_tr)
X_te = torch.tensor(X_te)
y_tr = torch.tensor(y_tr, dtype=torch.long)
y_te = torch.tensor(y_te, dtype=torch.long)

train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────── Define MLP ───────────────────
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

model = SimpleMLP(X_tr.shape[1], len(classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# weighted cross‑entropy
criterion = nn.CrossEntropyLoss(weight=cls_wts.to(device))
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─────────────────── Training ───────────────────
model.train()
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} – loss: {total_loss/len(train_ds):.4f}")

# ─────────────────── Evaluation ───────────────────
model.eval()
with torch.no_grad():
    logits = model(X_te.to(device))
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = y_te.numpy()

print("\n📊 Classification report:")
print(classification_report(y_true, y_pred, target_names=classes))
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Macro‑F1 :", f1_score(y_true, y_pred, average="macro"))

# ─────────────────── Save artefacts ───────────────────
os.makedirs("career_mlp", exist_ok=True)
torch.save(model.state_dict(),        "career_mlp/model.pth")
joblib.dump(le,                       "career_mlp/label_encoder.pkl")
joblib.dump(scaler,                   "career_mlp/scaler.pkl")
print("\n💾 Saved MLP model + encoder + scaler to career_mlp/")
