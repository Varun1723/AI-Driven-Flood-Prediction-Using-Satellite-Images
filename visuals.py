import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from tqdm import tqdm

# === Config ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Automatically get current script directory
DATA_DIR = os.path.join(BASE_DIR, "processed_dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "visualization_results")
MODEL_PATH = os.path.join(BASE_DIR, "flood_model.pth")  # Path to saved model

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Dataset class ===
class FloodDataset(Dataset):
    def __init__(self, folders):
        self.folders = folders

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        vv = np.load(os.path.join(folder, "VV.npy"))
        vh = np.load(os.path.join(folder, "VH.npy"))
        label = np.load(os.path.join(folder, "label.npy"))

        image = np.stack([vv, vh], axis=0).astype(np.float32)
        label = label.astype(np.float32)

        return torch.tensor(image), torch.tensor(label), folder

def pad_collate_fn(batch):
    import torch.nn.functional as F

    images, masks, folders = zip(*batch)

    # Get max height and width from all images
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    padded_images = []
    padded_masks = []

    for img, mask in zip(images, masks):
        # Pad image: shape [2, H, W]
        pad_h = max_height - img.shape[1]
        pad_w = max_width - img.shape[2]
        padded_img = F.pad(img, (0, pad_w, 0, pad_h))  # pad: (left, right, top, bottom)
        padded_images.append(padded_img)

        # Pad mask: shape [1, H, W] or [H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        pad_h_m = max_height - mask.shape[1]
        pad_w_m = max_width - mask.shape[2]
        padded_mask = F.pad(mask, (0, pad_w_m, 0, pad_h_m))
        padded_masks.append(padded_mask)

    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(padded_masks)
    
    return images_tensor, masks_tensor, folders

# === CNN Model ===
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
        return x

# === Visualization Functions ===

def visualize_sample_predictions(model, loader, num_samples=5):
    """Create visualization of model predictions vs ground truth"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Generating prediction visualizations for {num_samples} samples...")
    
    with torch.no_grad():
        for i, (inputs, targets, folders) in enumerate(loader):
            if i >= num_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            
            for j in range(len(inputs)):
                sample_name = os.path.basename(folders[j])
                
                # Extract inputs and outputs
                vv = inputs[j][0].cpu().numpy()
                vh = inputs[j][1].cpu().numpy()
                target = targets[j][0].cpu().numpy()
                output = outputs[j][0].cpu().numpy()
                pred = predictions[j][0].cpu().numpy()
                
                # Create visualization
                plt.figure(figsize=(15, 10))
                
                # First row: Input channels
                plt.subplot(2, 3, 1)
                plt.imshow(vv, cmap='gray')
                plt.title('VV Channel')
                plt.colorbar(fraction=0.046, pad=0.04)
                
                plt.subplot(2, 3, 2)
                plt.imshow(vh, cmap='gray')
                plt.title('VH Channel')
                plt.colorbar(fraction=0.046, pad=0.04)
                
                # Create a composite "false color" image
                plt.subplot(2, 3, 3)
                # Normalize for visualization
                vv_norm = (vv - vv.min()) / (vv.max() - vv.min())
                vh_norm = (vh - vh.min()) / (vh.max() - vh.min())
                composite = np.stack([vv_norm, vh_norm, (vv_norm+vh_norm)/2], axis=2)
                plt.imshow(composite)
                plt.title('Composite (VV-VH)')
                
                # Second row: Ground truth and predictions
                plt.subplot(2, 3, 4)
                plt.imshow(target, cmap='Blues')
                plt.title('Ground Truth')
                plt.colorbar(fraction=0.046, pad=0.04)
                
                plt.subplot(2, 3, 5)
                plt.imshow(output, cmap='Blues')
                plt.title('Model Output (Raw)')
                plt.colorbar(fraction=0.046, pad=0.04)
                
                plt.subplot(2, 3, 6)
                # Overlay prediction on ground truth
                overlay = np.zeros((target.shape[0], target.shape[1], 3))
                overlay[:,:,0] = pred  # Red channel: prediction
                overlay[:,:,2] = target  # Blue channel: ground truth
                overlay[:,:,1] = pred * target  # Green channel: overlap (correctly predicted flood)
                plt.imshow(overlay)
                plt.title('Overlay (Red=Pred, Blue=GT, Green=Correct)')
                
                # Add accuracy metrics
                accuracy = np.mean((pred == target).astype(np.float32))
                plt.suptitle(f'Sample: {sample_name}, Accuracy: {accuracy:.2%}', fontsize=16)
                
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f'prediction_{i+1}_{j}.png'), dpi=300)
                plt.close()
    
    print(f"✅ Saved prediction visualizations to {RESULTS_DIR}")

def visualize_dataset_examples(loader, num_samples=3):
    """Visualize examples from the dataset"""
    print(f"Generating dataset example visualizations...")
    
    for i, (inputs, targets, folders) in enumerate(loader):
        if i >= num_samples:
            break
            
        for j in range(len(inputs)):
            sample_name = os.path.basename(folders[j])
            
            vv = inputs[j][0].numpy()
            vh = inputs[j][1].numpy()
            label = targets[j][0].numpy()
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 4, 1)
            plt.imshow(vv, cmap='gray')
            plt.title('VV Channel')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.subplot(1, 4, 2)
            plt.imshow(vh, cmap='gray')
            plt.title('VH Channel')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.subplot(1, 4, 3)
            plt.imshow(label, cmap='Blues')
            plt.title('Flood Label')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # Create a ratio image (VV/VH)
            plt.subplot(1, 4, 4)
            # Avoid division by zero
            ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh!=0)
            # Clip for better visualization
            ratio = np.clip(ratio, 0, 5)
            plt.imshow(ratio, cmap='viridis')
            plt.title('VV/VH Ratio')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Dataset Sample: {sample_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'dataset_example_{i+1}_{j}.png'), dpi=300)
            plt.close()
    
    print(f"✅ Saved dataset examples to {RESULTS_DIR}")

def visualize_model_performance(model, loader, num_batches=10):
    """Generate performance metrics and visualizations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize metrics storage
    all_preds = []
    all_targets = []
    
    print("Computing performance metrics...")
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(tqdm(loader)):
            if i >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            
            # Store predictions and targets
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Flood', 'Flood'], 
                yticklabels=['No Flood', 'Flood'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'), dpi=300)
    plt.close()
    
    print(f"✅ Saved performance metrics to {RESULTS_DIR}")

def plot_VV_VH_correlation(loader, num_samples=5):
    """Analyze correlation between VV and VH channels for flooded vs non-flooded pixels"""
    print("Analyzing VV/VH correlation...")
    
    # Storage for pixel values
    flood_vv = []
    flood_vh = []
    nonflood_vv = []
    nonflood_vh = []
    
    for i, (inputs, targets, _) in enumerate(loader):
        if i >= num_samples:
            break
            
        for j in range(len(inputs)):
            vv = inputs[j][0].numpy().flatten()
            vh = inputs[j][1].numpy().flatten()
            mask = targets[j][0].numpy().flatten()
            
            # Sample pixels to prevent overwhelming the plot
            sample_size = min(1000, len(vv))
            indices = np.random.choice(len(vv), sample_size, replace=False)
            
            # Separate flooded and non-flooded pixels
            flood_pixels = mask > 0.5
            flood_vv.extend(vv[indices][flood_pixels[indices]])
            flood_vh.extend(vh[indices][flood_pixels[indices]])
            nonflood_vv.extend(vv[indices][~flood_pixels[indices]])
            nonflood_vh.extend(vh[indices][~flood_pixels[indices]])
    
    # Convert to numpy arrays
    flood_vv = np.array(flood_vv)
    flood_vh = np.array(flood_vh)
    nonflood_vv = np.array(nonflood_vv)
    nonflood_vh = np.array(nonflood_vh)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(nonflood_vv, nonflood_vh, c='gray', alpha=0.5, label='Non-Flooded', s=10)
    plt.scatter(flood_vv, flood_vh, c='blue', alpha=0.5, label='Flooded', s=10)
    plt.xlabel('VV Backscatter')
    plt.ylabel('VH Backscatter')
    plt.title('VV vs VH Backscatter for Flooded and Non-Flooded Areas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'vv_vh_correlation.png'), dpi=300)
    plt.close()
    
    # Create histograms
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(nonflood_vv, bins=50, alpha=0.5, color='gray', label='Non-Flooded')
    plt.hist(flood_vv, bins=50, alpha=0.5, color='blue', label='Flooded')
    plt.xlabel('VV Backscatter')
    plt.ylabel('Frequency')
    plt.title('VV Backscatter Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(nonflood_vh, bins=50, alpha=0.5, color='gray', label='Non-Flooded')
    plt.hist(flood_vh, bins=50, alpha=0.5, color='blue', label='Flooded')
    plt.xlabel('VH Backscatter')
    plt.ylabel('Frequency')
    plt.title('VH Backscatter Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Calculate VV/VH ratio (avoiding division by zero)
    flood_ratio = np.divide(flood_vv, flood_vh, out=np.zeros_like(flood_vv), where=flood_vh!=0)
    nonflood_ratio = np.divide(nonflood_vv, nonflood_vh, out=np.zeros_like(nonflood_vv), where=nonflood_vh!=0)
    # Clip for better visualization
    flood_ratio = np.clip(flood_ratio, 0, 5)
    nonflood_ratio = np.clip(nonflood_ratio, 0, 5)
    
    plt.hist(nonflood_ratio, bins=50, alpha=0.5, color='gray', label='Non-Flooded')
    plt.hist(flood_ratio, bins=50, alpha=0.5, color='blue', label='Flooded')
    plt.xlabel('VV/VH Ratio')
    plt.ylabel('Frequency')
    plt.title('VV/VH Ratio Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'backscatter_histograms.png'), dpi=300)
    plt.close()
    
    print(f"✅ Saved VV/VH correlation analysis to {RESULTS_DIR}")

def main():
    # Load data
    all_folders = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"Found {len(all_folders)} data folders")
    
    # Use a subset for analysis - adjust this as needed
    visualization_folders = all_folders[:30]  # Using 30 samples for visualization
    
    # Create dataset and loader
    dataset = FloodDataset(visualization_folders)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    
    # Load model
    model = FloodCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
        model_loaded = True
    except:
        print(f"Could not load model from {MODEL_PATH}. Will only run dataset visualizations.")
        model_loaded = False
    
    # Run visualizations
    visualize_dataset_examples(loader, num_samples=3)
    plot_VV_VH_correlation(loader, num_samples=5)
    
    if model_loaded:
        visualize_sample_predictions(model, loader, num_samples=5)
        visualize_model_performance(model, loader, num_batches=10)
    
    print(f"All visualizations completed and saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()