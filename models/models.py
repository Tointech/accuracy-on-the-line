"""
Model definitions and training utilities for experiments.
"""

from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_models() -> Dict[str, object]:
    """Return a dictionary of sklearn models for classification."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Linear Model": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Random Features": make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=1.0, n_components=500),
            LogisticRegression(max_iter=1000),
        ),
    }
    return models


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 32 * 3, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)


def train_mlp(
    X_train, y_train, X_test, y_test, X_102, y_102, device=DEVICE
) -> Tuple[float, float]:
    """Train a simple MLP on CIFAR-10 and evaluate on test and CIFAR-102 sets."""
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Convert data to tensors and create DataLoader for batching
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    X_102_t = torch.tensor(X_102, dtype=torch.float32).to(device)
    y_102_t = torch.tensor(y_102, dtype=torch.long).to(device)

    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).argmax(1)
        acc_test = (pred_test == y_test_t).float().mean().item()

        pred_102 = model(X_102_t).argmax(1)
        acc_102 = (pred_102 == y_102_t).float().mean().item()

    return acc_test, acc_102
