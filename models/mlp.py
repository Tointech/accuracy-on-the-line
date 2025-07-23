import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.model(x)


def train_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    X_102,
    y_102,
    hidden_dim=512,
    lr=1e-3,
    epochs=10,
    device=None,
):
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = MLP(hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    X_102_t = torch.tensor(X_102, dtype=torch.float32).to(device)
    y_102_t = torch.tensor(y_102, dtype=torch.long).to(device)
    for epoch in range(epochs):
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
