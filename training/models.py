"""
Deep Neural Network Architectures for Sign Language Landmark Classification
Includes:
- HandshapeMLP: Deep Residual Multi-Layer Perceptron for 3D Hand Landmark Configurations
- MovementSeqNet: Bidirectional Temporal GRU/CNN for Movement Stroke Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = F.leaky_relu(out + residual, negative_slope=0.1)
        return out

class HandshapeMLP(nn.Module):
    """
    Deep Residual Neural Network for 3D Normalized Hand Landmarks (21 points x 3 = 63 input dims)
    """
    def __init__(self, input_dim=63, hidden_dim=256, num_classes=14, dropout=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        self.res1 = ResBlock(hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, dropout)
        
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.fc_mid(out)
        logits = self.classifier(out)
        return logits


class MovementSeqNet(nn.Module):
    """
    Bidirectional GRU Temporal Sequence Classifier for Hand Movement Trajectories
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=12, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        # Global average pooling over time
        pooled = torch.mean(gru_out, dim=1)
        logits = self.fc(pooled)
        return logits
