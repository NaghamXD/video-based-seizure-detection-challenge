import numpy as np
import pandas as pd
import torch
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = 'train_data' # Update this!
LABEL_CSV = 'train_data/train_data.csv'         # Update this!
OUTPUT_FOLDER = 'processed_landmarks'

# Mapping 33 MediaPipe Landmarks to 15 VSViG/OpenPose-like Joints
# MediaPipe Indices:
# 0: Nose
# 2: Left Eye,       5: Right Eye (Using pupils/center of eye region)
# 11: Left Shoulder, 12: Right Shoulder
# 13: Left Elbow,    14: Right Elbow
# 15: Left Wrist,    16: Right Wrist
# 23: Left Hip,      24: Right Hip
# 25: Left Knee,     26: Right Knee
# 27: Left Ankle,    28: Right Ankle

# VSViG expects 15 points divided into 5 partitions (3 joints each):
# 1. Head: Nose, Left Eye, Right Eye
# 2. Right Arm: Right Wrist, Right Elbow, Right Shoulder
# 3. Right Leg: Right Hip, Right Knee, Right Ankle
# 4. Left Arm: Left Wrist, Left Elbow, Left Shoulder
# 5. Left Leg: Left Hip, Left Knee, Left Ankle

# Mapping dictionary: Target_Index -> MediaPipe_Index
# The order here must match the VSViG graph structure order.
# Assuming standard order (Nose, L-Eye, R-Eye, ... or similar)
# Let's map based on the paper description groups, but usually, indices are:
# 0:Nose, 1:LEye, 2:REye, 
# 3:RWri, 4:RElb, 5:RSho, 
# 6:RHip, 7:RKnee, 8:RAnk, 
# 9:LWri, 10:LElb, 11:LSho, 
# 12:LHip, 13:LKnee, 14:LAnk
# (Note: Exact index order depends on VSViG implementation, but this groups them logically)

MP_MAP = {
    # Partition 1: Head
    0: [0],       # Nose
    1: [2],       # Left Eye
    2: [5],       # Right Eye
    
    # Partition 2: Right Arm
    3: [12],      # Right Shoulder
    4: [14],      # Right Elbow
    5: [16],      # Right Wrist
    
    # Partition 3: Right Leg
    6: [24],      # Right Hip
    7: [26],      # Right Knee
    8: [28],      # Right Ankle
    
    # Partition 4: Left Arm
    9: [11],      # Left Shoulder
    10: [13],     # Left Elbow
    11: [15],     # Left Wrist
    
    # Partition 5: Left Leg
    12: [23],     # Left Hip
    13: [25],     # Left Knee
    14: [27],     # Left Ankle
}

def process_file_padding(npy_path):
    try:
        # Load raw data (N_frames, 33, 5)
        raw_lmks = np.load(npy_path)
    except:
        return None, None

    if raw_lmks.size == 0:
        return None, None

    # Create empty containers
    mapped_data = np.zeros((150, 15, 5), dtype=np.float32)
    valid_mask = np.zeros((150), dtype=bool)

    num_frames = min(150, raw_lmks.shape[0])
    if num_frames == 0: return None, None

    # Slice valid frames
    current_raw = raw_lmks[:num_frames].copy() # Copy to avoid modifying original if cached

    # --- NORMALIZATION STRATEGY ---
    # 1. Center X and Y (Avg is ~0.5, so we shift to 0.0)
    current_raw[:, :, 0] = current_raw[:, :, 0] - 0.5
    current_raw[:, :, 1] = current_raw[:, :, 1] - 0.5
    
    # 2. Scale X, Y, Z to range [-1, 1]
    # Max observed range is approx +/- 6.0. Dividing by 6.0 keeps data safe.
    # Note: Z is typically depth relative to hip, usually smaller range, but scaling consistently preserves aspect ratio.
    current_raw[:, :, 0:3] = current_raw[:, :, 0:3] / 6.0

    # 3. Mask Generation (Check for NaNs in original data)
    # We check before nan_to_num logic effectively, or just check the raw values
    # Actually, NaNs might be present. Let's check them.
    is_frame_nan = np.isnan(current_raw).all(axis=(1, 2))
    valid_mask[:num_frames] = ~is_frame_nan

    # 4. Map and Fill
    for target_idx, mp_indices in MP_MAP.items():
        source_joint_idx = mp_indices[0]
        source_data = current_raw[:, source_joint_idx, :] # (num_frames, 5)
        
        # Replace NaNs with 0.0 (which is now the center of image!)
        source_data = np.nan_to_num(source_data, nan=0.0)
        
        mapped_data[:num_frames, target_idx, :] = source_data

    return torch.from_numpy(mapped_data), torch.from_numpy(valid_mask)

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    print(f"📂 Loading labels from {LABEL_CSV}...")
    df = pd.read_csv(LABEL_CSV)
    
    processed_data_list = []
    processed_masks_list = []
    processed_labels_list = []
    processed_files = []
    
    print(f"🚀 Processing {len(df)} files with Zero-Padding strategy...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['segment_name'] # e.g. child_1_1.npy
        label = row['label']
        
        file_path = os.path.join(DATA_ROOT, filename)
        
        if not os.path.exists(file_path):
            # Optional: Print warning for missing files
            # print(f"Warning: File not found {filename}")
            continue
            
        data_tensor, mask_tensor = process_file_padding(file_path)
        
        if data_tensor is not None:
            processed_data_list.append(data_tensor)
            processed_masks_list.append(mask_tensor)
            processed_labels_list.append(label)
            processed_files.append(filename)
            
    print("💾 Stacking and Saving...")
    if len(processed_data_list) > 0:
        # Stack into big tensors
        all_data = torch.stack(processed_data_list)   # Shape: (N, 150, 15, 5)
        all_masks = torch.stack(processed_masks_list) # Shape: (N, 150)
        all_labels = torch.tensor(processed_labels_list)
        
        # Save mapping of index -> filename for submission generation
        pd.DataFrame({'file_name': processed_files}).to_csv(os.path.join(OUTPUT_FOLDER, 'file_map.csv'), index=False)
        
        torch.save(all_data, os.path.join(OUTPUT_FOLDER, 'all_landmarks.pt'))
        torch.save(all_masks, os.path.join(OUTPUT_FOLDER, 'all_masks.pt'))
        torch.save(all_labels, os.path.join(OUTPUT_FOLDER, 'all_labels.pt'))
        
        print(f"✅ Successfully saved {len(processed_data_list)} samples.")
        print(f"   Data Tensor Shape: {all_data.shape}")
        print(f"   Mask Tensor Shape: {all_masks.shape}")
        print(f"   Labels Shape: {all_labels.shape}")
    else:
        print("❌ No valid data found. Check paths.")

if __name__ == '__main__':
    main()