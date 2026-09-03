"""
Training Pipeline for Neural Movement & Gesture Sequence Network (MovementSeqNet)
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from models import MovementSeqNet

CLASSES = [
    "hamnomotion",
    "hammoveu",
    "hammoved",
    "hammovel",
    "hammover",
    "hammoveui",
    "hammoveuo",
    "hammovedi",
    "hammovedo",
    "hamwavy",
    "hamzigzag",
    "hamcircleo",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def generate_movement_sequences(num_samples_per_class=800, seq_len=32):
    """
    Generates dynamic 3D trajectory sequences (displacement + velocity = 6 dims)
    for HamNoSys movement primitives.
    """
    X_list = []
    y_list = []

    t = np.linspace(0, 1, seq_len)

    for class_name in CLASSES:
        c_idx = CLASS_TO_IDX[class_name]
        for _ in range(num_samples_per_class):
            # Base trajectory (x, y, z)
            speed = np.random.uniform(0.7, 1.3)
            amp = np.random.uniform(0.15, 0.35)
            
            if class_name == "hamnomotion":
                x = np.zeros(seq_len)
                y = np.zeros(seq_len)
                z = np.zeros(seq_len)
            elif class_name == "hammoveu":
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = -amp * (t * speed)
                z = np.random.uniform(-0.02, 0.02, seq_len)
            elif class_name == "hammoved":
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = amp * (t * speed)
                z = np.random.uniform(-0.02, 0.02, seq_len)
            elif class_name == "hammovel":
                x = -amp * (t * speed)
                y = np.random.uniform(-0.02, 0.02, seq_len)
                z = np.random.uniform(-0.02, 0.02, seq_len)
            elif class_name == "hammover":
                x = amp * (t * speed)
                y = np.random.uniform(-0.02, 0.02, seq_len)
                z = np.random.uniform(-0.02, 0.02, seq_len)
            elif class_name == "hammoveui": # Up + Towards signer
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = -amp * (t * speed)
                z = amp * 0.7 * (t * speed)
            elif class_name == "hammoveuo": # Up + Outwards
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = -amp * (t * speed)
                z = -amp * 0.7 * (t * speed)
            elif class_name == "hammovedi": # Down + Towards signer
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = amp * (t * speed)
                z = amp * 0.7 * (t * speed)
            elif class_name == "hammovedo": # Down + Outwards
                x = np.random.uniform(-0.02, 0.02, seq_len)
                y = amp * (t * speed)
                z = -amp * 0.7 * (t * speed)
            elif class_name == "hamwavy":
                x = -amp * t
                y = 0.08 * np.sin(2 * math.pi * 2.5 * t)
                z = np.zeros(seq_len)
            elif class_name == "hamzigzag":
                x = -amp * t
                y = 0.08 * (2 * np.abs(2 * (2.5 * t - np.floor(2.5 * t + 0.5))) - 1)
                z = np.zeros(seq_len)
            else: # hamcircleo
                angle = 2 * math.pi * t * speed
                x = amp * 0.5 * np.cos(angle)
                y = amp * 0.5 * np.sin(angle)
                z = np.zeros(seq_len)

            # Add noise and jitter
            noise = np.random.normal(0, 0.01, size=(seq_len, 3))
            pos = np.stack([x, y, z], axis=1) + noise
            
            # Compute velocities (dx, dy, dz)
            vel = np.gradient(pos, axis=0)

            # Combined feature representation (6 features per timestep)
            features = np.concatenate([pos, vel], axis=1).astype(np.float32)

            X_list.append(features)
            y_list.append(c_idx)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_movement_model():
    print("Generating temporal movement sequences across 12 HamNoSys motion classes...")
    X, y = generate_movement_sequences(num_samples_per_class=1000, seq_len=32)
    print(f"Total temporal movement samples: {len(X)} with shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = MovementDataset(X_train, y_train)
    val_dataset = MovementDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = MovementSeqNet(input_dim=6, hidden_dim=64, num_layers=2, num_classes=len(CLASSES), dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    best_val_acc = 0.0
    output_dir = Path(__file__).resolve().parent.parent / "Integration-20260706T062240Z-3-001" / "Integration"
    model_path = output_dir / "nn_movement_model.pt"

    print(f"\nTraining MovementSeqNet for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_targets_all.extend(batch_y.cpu().numpy())

        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "input_dim": 6,
                "hidden_dim": 64,
                "num_layers": 2,
                "val_acc": val_acc
            }, model_path)

    print(f"\n[Training Complete] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved neural model weights to: {model_path.resolve()}")

    print("\nDetailed Validation Classification Report:")
    print(classification_report(val_targets_all, val_preds_all, target_names=CLASSES, digits=4))

if __name__ == "__main__":
    train_movement_model()
