import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "processed_dataset"
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

# --- Dataset Class ---
class FloodDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        vv = np.load(os.path.join(folder, "VV.npy"))
        vh = np.load(os.path.join(folder, "VH.npy"))
        label = np.load(os.path.join(folder, "label.npy"))

        min_h = min(vv.shape[0], vh.shape[0], label.shape[0])
        min_w = min(vv.shape[1], vh.shape[1], label.shape[1])

        vv = vv[:min_h, :min_w]
        vh = vh[:min_h, :min_w]
        label = label[:min_h, :min_w]

        vv = torch.tensor(vv, dtype=torch.float32)
        vh = torch.tensor(vh, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        image = torch.stack([vv, vh], dim=0)  # (2, H, W)
        return image, label  # label shape (H, W)

# --- Collate Function to Pad Tensors ---
def pad_collate_fn(batch):
    images, labels = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    padded_labels = []
    for img, lbl in zip(images, labels):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_lbl = F.pad(lbl, (0, pad_w, 0, pad_h), mode='constant', value=0)

        padded_images.append(padded_img)
        padded_labels.append(padded_lbl)

    return torch.stack(padded_images), torch.stack(padded_labels)

# --- Simple CNN Model ---
class FloodCNN(nn.Module):
    def __init__(self):
        super(FloodCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

# --- Training Function ---
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        targets = F.interpolate(targets.unsqueeze(1), size=outputs.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- Evaluation Function ---
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            targets = F.interpolate(targets.unsqueeze(1), size=outputs.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
            total_correct += (preds == targets).sum().item()
            total_pixels += torch.numel(preds)
    return total_correct / total_pixels

# --- Main Execution ---
def main():
    all_folders = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"✅ Total folders found: {len(all_folders)}")

    # Split data
    train_val_folders, test_folders = train_test_split(all_folders, test_size=0.2, random_state=42)
    train_folders, val_folders = train_test_split(train_val_folders, test_size=0.25, random_state=42)  # 60/20/20
    print(f"Train={len(train_folders)}, Val={len(val_folders)}, Test={len(test_folders)}")

    # Limit to ~2600 batches
    max_train_samples = BATCH_SIZE * 2600  # = 20800
    if len(train_folders) > max_train_samples:
        train_folders = train_folders[:max_train_samples]

    train_dataset = FloodDataset(train_folders)
    val_dataset = FloodDataset(val_folders)
    test_dataset = FloodDataset(test_folders)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    model = FloodCNN().cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    test_acc = evaluate(model, test_loader)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
