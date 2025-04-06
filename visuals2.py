import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FloodDataset(Dataset):
    def __init__(self, root_dir):
        self.data_folders = [
            os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name)) and
               os.path.exists(os.path.join(root_dir, name, "VV.npy")) and
               os.path.exists(os.path.join(root_dir, name, "VH.npy")) and
               os.path.exists(os.path.join(root_dir, name, "label.npy"))
        ]

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        folder = self.data_folders[idx]
        vv = np.load(os.path.join(folder, "VV.npy"))
        vh = np.load(os.path.join(folder, "VH.npy"))
        label = np.load(os.path.join(folder, "label.npy"))

        vv_tensor = torch.tensor(vv, dtype=torch.float32).unsqueeze(0)
        vh_tensor = torch.tensor(vh, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        input_tensor = torch.cat([vv_tensor, vh_tensor], dim=0)
        return input_tensor, label_tensor, folder


def pad_collate_fn(batch):
    inputs, targets, folders = zip(*batch)
    max_height = max(t.shape[1] for t in inputs)
    max_width = max(t.shape[2] for t in inputs)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        pad_h = max_height - inp.shape[1]
        pad_w = max_width - inp.shape[2]
        padded_inp = F.pad(inp, (0, pad_w, 0, pad_h))
        padded_tgt = F.pad(tgt, (0, pad_w, 0, pad_h))
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)

    input_batch = torch.stack(padded_inputs)
    target_batch = torch.stack(padded_targets)

    return input_batch, target_batch, folders
