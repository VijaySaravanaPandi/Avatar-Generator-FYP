"""
Training Pipeline for Deep Neural Handshape Classifier
"""

import os
import sys
import pickle
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from models import HandshapeMLP

# Canonical HamNoSys Handshape Classes
CLASSES = [
    "hamflathand",
    "hamfist",
    "hamfinger2",
    "hamfinger23",
    "hamfinger2345",
    "hamfinger23spread",
    "hampinch12",
    "hampinchall",
    "hamcee12",
    "hamceeall",
    "hamdoublebent",
    "hamthumboutmod",
    "hamthumbopenmod",
    "hamthumbacrossmod",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def augment_landmarks(landmarks_63):
    """Applies random 3D rotations, scaling, and jitter to landmarks."""
    pts = landmarks_63.reshape(21, 3).copy()
    
    # Random small rotation angle in radians
    angle_x = np.random.uniform(-0.25, 0.25)
    angle_y = np.random.uniform(-0.25, 0.25)
    angle_z = np.random.uniform(-0.35, 0.35)
    
    Rx = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]])
    Ry = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]])
    Rz = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]])
    
    R = Rx @ Ry @ Rz
    pts = pts @ R.T
    
    # Slight jitter
    noise = np.random.normal(0, 0.015, size=pts.shape)
    pts += noise
    
    # Re-normalize
    max_d = np.max(np.linalg.norm(pts, axis=1))
    if max_d > 1e-6:
        pts = pts / max_d
        
    return pts.flatten()

def generate_canonical_handshape_dataset(num_samples_per_class=1200):
    """
    Generates a rich, high-diversity training dataset representing all 14 HamNoSys
    handshape classes with anatomical hand constraints and full 3D spatial augmentations.
    """
    X_list = []
    y_list = []

    # Base 21 landmark topology template
    base_joints = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    base_joints[0] = [0.0, 0.0, 0.0]
    
    # Metacarpals
    base_joints[1] = [-0.15, 0.15, 0.0] # Thumb CMC
    base_joints[5] = [-0.10, 0.40, 0.0] # Index MCP
    base_joints[9] = [0.0, 0.42, 0.0]   # Middle MCP
    base_joints[13] = [0.08, 0.38, 0.0] # Ring MCP
    base_joints[17] = [0.15, 0.32, 0.0] # Pinky MCP

    def make_finger(mcp_idx, base_dir, curl_amount, spread=0.0):
        # Generates PIP, DIP, TIP from MCP
        mcp = base_joints[mcp_idx]
        curled = 1.0 - curl_amount
        pip = mcp + np.array([spread * 0.3, 0.20 * curled, -0.15 * curl_amount])
        dip = pip + np.array([spread * 0.4, 0.18 * curled, -0.18 * curl_amount])
        tip = dip + np.array([spread * 0.3, 0.16 * curled, -0.15 * curl_amount])
        return pip, dip, tip

    for class_name in CLASSES:
        c_idx = CLASS_TO_IDX[class_name]
        for _ in range(num_samples_per_class):
            joints = base_joints.copy()
            
            # Curl states per finger (0.0 = fully open/extended, 1.0 = fully curled/fist)
            if class_name == "hamflathand":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.1, 0.0, 0.0, 0.0, 0.0
                t_spread = -0.2
            elif class_name == "hamfist":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.9, 0.95, 0.95, 0.95, 0.95
                t_spread = 0.0
            elif class_name == "hamfinger2":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.8, 0.0, 0.95, 0.95, 0.95
                t_spread = 0.0
            elif class_name == "hamfinger23":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.8, 0.0, 0.0, 0.95, 0.95
                t_spread = 0.0
            elif class_name == "hamfinger23spread":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.8, 0.0, 0.0, 0.95, 0.95
                t_spread = -0.15
            elif class_name == "hamfinger2345":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.85, 0.0, 0.0, 0.0, 0.0
                t_spread = 0.0
            elif class_name == "hampinch12":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.5, 0.6, 0.9, 0.9, 0.9
                t_spread = 0.05
            elif class_name == "hampinchall":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.6, 0.6, 0.6, 0.6, 0.6
                t_spread = 0.0
            elif class_name == "hamcee12":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.35, 0.35, 0.9, 0.9, 0.9
                t_spread = -0.1
            elif class_name == "hamceeall":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.35, 0.35, 0.35, 0.35, 0.35
                t_spread = -0.1
            elif class_name == "hamdoublebent":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.6, 0.5, 0.5, 0.9, 0.9
                t_spread = 0.0
            elif class_name == "hamthumboutmod":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.0, 0.9, 0.9, 0.9, 0.9
                t_spread = -0.4
            elif class_name == "hamthumbopenmod":
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.0, 0.0, 0.0, 0.0, 0.0
                t_spread = -0.35
            else: # hamthumbacrossmod
                t_curl, i_curl, m_curl, r_curl, p_curl = 0.85, 0.0, 0.0, 0.0, 0.0
                t_spread = 0.2

            # Build joints
            # Thumb: 1(CMC), 2(MCP), 3(IP), 4(TIP)
            joints[2] = joints[1] + np.array([t_spread * 0.5, 0.15 * (1.0 - t_curl), -0.1 * t_curl])
            joints[3] = joints[2] + np.array([t_spread * 0.5, 0.14 * (1.0 - t_curl), -0.1 * t_curl])
            joints[4] = joints[3] + np.array([t_spread * 0.4, 0.12 * (1.0 - t_curl), -0.1 * t_curl])
            
            # Index (5, 6, 7, 8)
            joints[6], joints[7], joints[8] = make_finger(5, [0, 1, 0], i_curl)
            # Middle (9, 10, 11, 12)
            joints[10], joints[11], joints[12] = make_finger(9, [0, 1, 0], m_curl)
            # Ring (13, 14, 15, 16)
            joints[14], joints[15], joints[16] = make_finger(13, [0, 1, 0], r_curl)
            # Pinky (17, 18, 19, 20)
            joints[18], joints[19], joints[20] = make_finger(17, [0, 1, 0], p_curl)

            # Center & normalize
            max_d = np.max(np.linalg.norm(joints, axis=1))
            norm_feats = (joints / max_d).flatten()
            
            # Apply augmentation
            aug_feats = augment_landmarks(norm_feats)
            X_list.append(aug_feats)
            y_list.append(c_idx)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

class HandshapeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model():
    print("Generating comprehensive 3D handshape dataset across all 14 HamNoSys classes...")
    X, y = generate_canonical_handshape_dataset(num_samples_per_class=1500)
    print(f"Total training samples: {len(X)} with 63-dimensional normalized 3D features.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = HandshapeDataset(X_train, y_train)
    val_dataset = HandshapeDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = HandshapeMLP(input_dim=63, hidden_dim=256, num_classes=len(CLASSES), dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    epochs = 25
    best_val_acc = 0.0
    output_dir = Path(__file__).resolve().parent.parent / "Integration-20260706T062240Z-3-001" / "Integration"
    model_path = output_dir / "nn_handshape_model.pt"

    print(f"\nBeginning training for {epochs} epochs...")
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
                "input_dim": 63,
                "hidden_dim": 256,
                "val_acc": val_acc
            }, model_path)

    print(f"\n[Training Complete] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved neural model weights to: {model_path.resolve()}")

    # Print final detailed classification report
    print("\nDetailed Validation Classification Report:")
    print(classification_report(val_targets_all, val_preds_all, target_names=CLASSES, digits=4))

if __name__ == "__main__":
    train_model()
