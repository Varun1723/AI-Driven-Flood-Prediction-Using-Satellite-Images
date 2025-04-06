import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from visuals2 import FloodDataset, pad_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FloodCNN(nn.Module):
    def __init__(self, num_filters=16, dropout=0.2):
        super(FloodCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(num_filters, num_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters * 2),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),

            nn.ConvTranspose2d(num_filters, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            outputs = outputs.squeeze(1)

            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=1)
    return acc, f1


def train_with_hyperparams(params):
    # Hyperparameters
    lr = params.get("lr", 0.001)
    batch_size = params.get("batch_size", 16)
    num_filters = params.get("num_filters", 16)
    dropout = params.get("dropout", 0.2)
    epochs = params.get("epochs", 3)  # Short for speed

    # Load dataset
    dataset = FloodDataset("processed_dataset/")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    model = FloodCNN(num_filters=num_filters, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(epochs):
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    acc, f1 = evaluate_model(model, val_loader)
    print(f"[Eval] Acc: {acc:.4f}, F1: {f1:.4f}")
    return f1  # Fitness = F1 score


if __name__ == "__main__":
    params = {
        "lr": 0.001,
        "batch_size": 8,
        "num_filters": 16,
        "dropout": 0.3,
        "epochs": 2
    }
    f1 = train_with_hyperparams(params)
    print("Fitness (F1):", f1)
