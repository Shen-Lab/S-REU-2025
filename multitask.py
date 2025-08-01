import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- Dataset ----------------------
class MutationDataset(Dataset):
    def __init__(self, embeddings, dms_scores, coarse_types, pathogenicity_labels):
        self.embeddings = embeddings
        self.dms_scores = dms_scores
        self.coarse_types = coarse_types
        self.pathogenicity_labels = pathogenicity_labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.dms_scores[idx], dtype=torch.float32),
            self.coarse_types[idx],
            torch.tensor(self.pathogenicity_labels[idx], dtype=torch.float32)
        )

# ---------------------- Model ----------------------
class MultitaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.coarse_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1) for name in ['Activity', 'Binding', 'Expression', 'Stability', 'OrganismalFitness']
        })

        self.pathogenicity_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        shared = self.shared(x)
        dms_out = {k: head(shared).squeeze(-1) for k, head in self.coarse_heads.items()}
        patho_out = self.pathogenicity_head(shared)  # logits only
        return dms_out, patho_out

# ---------------------- Utils ----------------------
def encode_pathogenicity(label):
    classes = ['Pathogenic', 'Benign', 'Uncertain significance']
    vec = [1 if label == c else 0 for c in classes]
    return vec if sum(vec) else [0, 0, 0]

def prepare_data(data_dir, subs_file, patho_file, max_files=None):
    all_embeddings = []
    all_dms_scores = []
    all_coarse_types = []
    all_patho_labels = []

    subs_df = pd.read_csv(subs_file)
    patho_df = pd.read_csv(patho_file)
    patho_dict = dict(zip(patho_df['Mutation'], patho_df['ClinicalSignificance']))

    count = 0
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and file != os.path.basename(subs_file) and file != os.path.basename(patho_file):
            base_name = file
            csv_path = os.path.join(data_dir, base_name)
            npy_path = os.path.join(data_dir, base_name.replace('.csv', '_combined.npy'))

            if not os.path.exists(npy_path):
                print(f"Missing embedding file for {base_name}, skipping.")
                continue

            csv_df = pd.read_csv(csv_path)
            embeddings = np.load(npy_path)

            if len(embeddings) != len(csv_df):
                print(f"Mismatch in rows for {base_name}, skipping.")
                continue

            match = subs_df[subs_df['DMS_filename'] == base_name]
            if match.empty:
                print(f"Missing coarse type for {base_name}, skipping.")
                continue
            coarse_type = match['coarse_selection_type'].values[0]

            for i, row in csv_df.iterrows():
                mut_id = f"{base_name}:{row['mutant']}"
                dms_score = row['DMS_score']
                patho_vec = encode_pathogenicity(patho_dict.get(mut_id, ""))

                if sum(patho_vec) == 0:
                    continue  # skip unknown labels

                norm_emb = embeddings[i] / np.linalg.norm(embeddings[i])  # normalize
                all_embeddings.append(norm_emb)
                all_dms_scores.append(dms_score)
                all_coarse_types.append(coarse_type)
                all_patho_labels.append(patho_vec)

            count += 1
            if max_files and count >= max_files:
                break

    return all_embeddings, all_dms_scores, all_coarse_types, all_patho_labels

# ---------------------- Training ----------------------
def train_model(dataset, epochs=200, batch_size=64, lr=5e-4):
    train_set, test_val = train_test_split(dataset, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(test_val, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    input_dim = dataset[0][0].shape[0]
    model = MultitaskMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, dms_y, coarse_type, patho_y in train_loader:
            optimizer.zero_grad()
            dms_preds, patho_preds = model(x)

            losses = []
            for i in range(len(coarse_type)):
                ct = coarse_type[i]
                loss = criterion_reg(dms_preds[ct][i], dms_y[i])
                losses.append(loss)
            dms_loss = torch.stack(losses).mean()
            cls_loss = criterion_cls(patho_preds, patho_y.argmax(dim=1))
            total_loss = dms_loss * 0.5 + cls_loss * 0.5  # balance

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, dms_y, coarse_type, patho_y in val_loader:
                dms_preds, patho_preds = model(x)
                losses = []
                for i in range(len(coarse_type)):
                    ct = coarse_type[i]
                    loss = criterion_reg(dms_preds[ct][i], dms_y[i])
                    losses.append(loss)
                dms_loss = torch.stack(losses).mean()
                cls_loss = criterion_cls(patho_preds, patho_y.argmax(dim=1))
                total_loss = dms_loss * 0.5 + cls_loss * 0.5
                val_loss += total_loss.item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history.append({'epoch': epoch + 1, 'train_loss': avg_train, 'val_loss': avg_val})
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    return model, test_loader, history

# ---------------------- Evaluation ----------------------
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader):
    model.eval()

    # For overall DMS Spearman
    dms_true_all = []
    dms_pred_all = []

    # Per coarse type
    dms_per_type = {
        'Activity': {'true': [], 'pred': []},
        'Binding': {'true': [], 'pred': []},
        'Expression': {'true': [], 'pred': []},
        'Stability': {'true': [], 'pred': []},
        'OrganismalFitness': {'true': [], 'pred': []}
    }

    # Pathogenicity classification
    y_true_cls = []
    y_pred_cls = []

    with torch.no_grad():
        for x, dms_y, coarse_type, patho_y in test_loader:
            dms_preds, patho_logits = model(x)

            for i in range(len(coarse_type)):
                ct = coarse_type[i]
                true_val = dms_y[i].item()
                pred_val = dms_preds[ct][i].item()

                dms_true_all.append(true_val)
                dms_pred_all.append(pred_val)

                if ct in dms_per_type:
                    dms_per_type[ct]['true'].append(true_val)
                    dms_per_type[ct]['pred'].append(pred_val)

                y_true_cls.append(patho_y[i].argmax().item())
                y_pred_cls.append(patho_logits[i].argmax().item())

    # Output Spearman per type
    print("\n📊 Spearman Correlation per Coarse Selection Type:")
    for ct, values in dms_per_type.items():
        if len(values['true']) > 1:
            rho, _ = spearmanr(values['true'], values['pred'])
            print(f"  {ct}: ρ = {rho:.4f} ({len(values['true'])} samples)")
        else:
            print(f"  {ct}: not enough data.")

    # Overall Spearman
    rho_all, _ = spearmanr(dms_true_all, dms_pred_all)
    print(f"\n🔬 Overall DMS Spearman: ρ = {rho_all:.4f}")

    # Pathogenicity accuracy
    acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f"🧬 Pathogenicity Accuracy: {acc*100:.2f}%")

# ---------------------- Main ----------------------
def main():
    data_dir = '../../MultimodalStuff/finalem'
    subs_file = '../Structure-informed_PLM/mastercsv/DMS_substitutions.csv'
    patho_file = '../Structure-informed_PLM/mastercsv/mapped_pathogenicity_threeLabels.csv'
    
    max_files = 200
    epochs = 200
    lr = 5e-4

    print("Loading data...")
    embeddings, dms_scores, coarse_types, patho_labels = prepare_data(
        data_dir, subs_file, patho_file, max_files=max_files
    )

    # Normalize DMS scores
    dms_scores = np.array(dms_scores)
    dms_mean = dms_scores.mean()
    dms_std = dms_scores.std()
    dms_scores = (dms_scores - dms_mean) / dms_std

    dataset = MutationDataset(embeddings, dms_scores, coarse_types, patho_labels)

    print(f"Training model for {epochs} epochs at LR={lr}...")
    model, test_loader, history = train_model(dataset, epochs=epochs, lr=lr)

    print("Saving training log to training_log_improvedtest.csv...")
    pd.DataFrame(history).to_csv("training_log_improvedtest.csv", index=False)

    print("Evaluating model...")
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
