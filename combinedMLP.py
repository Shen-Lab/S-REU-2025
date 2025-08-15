import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
from scipy.stats import spearmanr
import csv
import pandas as pd

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directories
embedding_dir = "finalem"
label_dir = "esm2em"

embedding_files = sorted(f for f in os.listdir(embedding_dir) if f.endswith("_combined.npy"))

# Collect all data
all_embeddings = []
all_labels = []

for emb_file in embedding_files:
    assay_name = emb_file.replace("_combined.npy", "")
    emb_path = os.path.join(embedding_dir, emb_file)
    label_path = os.path.join(label_dir, f"{assay_name}_labels.npy")

    if not os.path.exists(label_path):
        print(f"⚠️ Skipping {assay_name}, no label file found.")
        continue

    embeddings = np.load(emb_path)
    labels = np.load(label_path).squeeze()

    if len(embeddings) != len(labels):
        print(f"⚠️ Skipping {assay_name}, mismatched lengths.")
        continue

    if len(labels) < 10:
        print(f"⚠️ Skipping {assay_name}, too few samples.")
        continue

    # ✅ Normalize labels (z-score) per assay
    mean = labels.mean()
    std = labels.std()
    if std == 0:
        print(f"⚠️ Skipping {assay_name}, label std = 0 (constant labels)")
        continue
    labels = (labels - mean) / std

    all_embeddings.append(embeddings)
    all_labels.append(labels)

# Combine all data
if not all_embeddings:
    raise ValueError("❌ No valid data found!")

X = torch.tensor(np.concatenate(all_embeddings, axis=0), dtype=torch.float32)
y = torch.tensor(np.concatenate(all_labels, axis=0), dtype=torch.float32)

# Dataset split
dataset = TensorDataset(X, y)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# 🧠 Model with Dropout
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.model(x)

# Initialize model
model = SimpleMLP(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Train loop
best_val_loss = float("inf")
training_losses = []
validation_losses = []

for epoch in range(200):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).reshape(-1)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).reshape(-1)
            val_loss += loss_fn(preds, yb).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    training_losses.append(avg_train_loss)
    validation_losses.append(avg_val_loss)

    print(f"📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

# Load best model
model.load_state_dict(best_model_state)
model.eval()
torch.save(model.state_dict(), "trained_global_model.pt")
print("💾 Model saved as 'trained_global_model.pt'")

with open("test2-2log.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for i, (train, val) in enumerate(zip(training_losses, validation_losses), start=1):
        writer.writerow([i, train, val])

print("📄 Training losses saved to 'test2-2log.csv'")

# Evaluate on test set
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).reshape(-1)
        all_preds.extend(preds.cpu().numpy().ravel())
        all_targets.extend(yb.cpu().numpy().ravel())

if len(set(all_targets)) > 1:
    spearman_corr = spearmanr(all_preds, all_targets).correlation
    print(f"\n✅ Spearman Correlation on Global Test Set: {spearman_corr:.4f}")
    print(f"🔍 Sample Predictions:")
    for i in range(min(3, len(all_preds))):
        print(f"  Prediction: {all_preds[i]:.4f} | True Label: {all_targets[i]:.4f}")
else:
    print("\n⚠️ Spearman skipped: constant labels in test set.")