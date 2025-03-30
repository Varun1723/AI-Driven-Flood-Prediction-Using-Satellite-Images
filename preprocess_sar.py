import os
import numpy as np
import rasterio  # Read .tif files
from tqdm import tqdm  # Progress bar

# ✅ Set correct dataset path - notice the change here
SAR_DIR = "C:/Varun/Coding/Flood_Prediction/sen12flood"
OUTPUT_DIR = "C:/Varun/Coding/Flood_Prediction/processed_data"

# ✅ Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Look specifically for date folders that contain the SAR images
sar_folders = []
s1_source_dir = os.path.join(SAR_DIR, "sen12floods_s1_source")

# Gather all the date folders that contain the actual VH/VV files
for root, dirs, files in os.walk(s1_source_dir):
    if "VH.tif" in files and "VV.tif" in files:
        sar_folders.append(root)

print(f"Found {len(sar_folders)} SAR image folders")

# ✅ Process each SAR image
num_processed = 0  # Counter

for folder_path in tqdm(sar_folders, desc="Processing SAR Images"):
    # Extract folder name for saving
    folder_name = os.path.basename(folder_path)
    
    # ✅ Define file paths
    vh_path = os.path.join(folder_path, "VH.tif")
    vv_path = os.path.join(folder_path, "VV.tif")

    # ✅ Check if SAR images exist (redundant but safe)
    if not os.path.exists(vh_path) or not os.path.exists(vv_path):
        print(f"Skipping {folder_name}, missing VH/VV files")
        continue

    # ✅ Read SAR images
    with rasterio.open(vh_path) as vh_src, rasterio.open(vv_path) as vv_src:
        vh = vh_src.read(1).astype(np.float32)  # Read band 1
        vv = vv_src.read(1).astype(np.float32)

    # ✅ Normalize values (convert to range [0,1])
    vh = (vh - vh.min()) / (vh.max() - vh.min() + 1e-6)
    vv = (vv - vv.min()) / (vv.max() - vv.min() + 1e-6)

    # ✅ Stack VH & VV into a single array (2 channels)
    sar_image = np.stack([vh, vv], axis=-1)

    # ✅ Save as .npy file
    save_path = os.path.join(OUTPUT_DIR, f"{folder_name}.npy")
    np.save(save_path, sar_image)
    num_processed += 1

print(f"\n✅ Preprocessing Complete! Processed {num_processed} images.")
print(f"Saved files to: {OUTPUT_DIR}")