import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from VSViG_Landmark import VSViG_Landmark_Base
from pathlib import Path
# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent

# --- CONFIGURATION ---
DATA_ROOT =  BASE_DIR /'train_data'            
SPLIT_FOLDER =  BASE_DIR /'processed_landmarks/splits_stratified' 
MAX_FRAMES = 150                    
BATCH_SIZE = 32
CHECKPOINT_DIR = BASE_DIR / "checkpoints_stratified"

# Mapping 33 MediaPipe Landmarks to 15 VSViG Joints (Must match Preprocessing logic)
MP_MAP = {
    0: [0],       # Nose
    1: [2],       # Left Eye
    2: [5],       # Right Eye
    3: [12],      # Right Shoulder
    4: [14],      # Right Elbow
    5: [16],      # Right Wrist
    6: [24],      # Right Hip
    7: [26],      # Right Knee
    8: [28],      # Right Ankle
    9: [11],      # Left Shoulder
    10: [13],     # Left Elbow
    11: [15],     # Left Wrist
    12: [23],     # Left Hip
    13: [25],     # Left Knee
    14: [27],     # Left Ankle
}

if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

class SeizureDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_frames=150):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get File Path and Label
        row = self.annotations.iloc[index]
        filename = row['segment_name']
        label = int(row['label'])
        
        file_path = os.path.join(self.root_dir, filename)

        # 2. Load Data
        try:
            # Raw Shape: (N_Frames, 33, 5)
            raw_lmks = np.load(file_path)
            if raw_lmks.size == 0:
                raw_lmks = np.zeros((1, 33, 5), dtype=np.float32)
        except Exception:
            raw_lmks = np.zeros((1, 33, 5), dtype=np.float32)

        # 3. Data Processing & Mapping (33 -> 15)
        # Create containers
        mapped_data = np.zeros((self.max_frames, 15, 5), dtype=np.float32)
        mask = np.zeros((self.max_frames), dtype=np.float32) # Float for interpolation

        num_frames = min(self.max_frames, raw_lmks.shape[0])
        
        if num_frames > 0:
            # Slice valid frames
            current_raw = raw_lmks[:num_frames].copy()

            # --- NORMALIZATION (Matches preprocessing_landmarks.py) ---
            # Center X and Y (-0.5) and Scale (/6.0)
            current_raw[:, :, 0] = current_raw[:, :, 0] - 0.5
            current_raw[:, :, 1] = current_raw[:, :, 1] - 0.5
            current_raw[:, :, 0:3] = current_raw[:, :, 0:3] / 6.0

            # --- MAPPING 33 -> 15 ---
            for target_idx, mp_indices in MP_MAP.items():
                source_idx = mp_indices[0]
                mapped_data[:num_frames, target_idx, :] = current_raw[:, source_idx, :]

            # --- MASK CREATION ---
            # 1 = Valid Frame, 0 = Padding
            mask[:num_frames] = 1.0 
            
            # Handle NaNs (Replace with 0.0)
            mapped_data = np.nan_to_num(mapped_data, nan=0.0)

        # 4. Convert to Tensor
        tensor_data = torch.from_numpy(mapped_data) # (150, 15, 5)
        tensor_label = torch.tensor(label, dtype=torch.float32) # Float for BCE
        tensor_mask = torch.from_numpy(mask) # (150)

        return tensor_data, tensor_label, tensor_mask

def get_dataloaders(split_dir, data_root, batch_size=32):
    train_csv = os.path.join(split_dir, 'train_split.csv')
    val_csv   = os.path.join(split_dir, 'val_split.csv')
    #test_csv  = os.path.join(split_dir, 'test_split.csv') # Load Test CSV
    
    train_dataset = SeizureDataset(train_csv, data_root, MAX_FRAMES)
    val_dataset   = SeizureDataset(val_csv, data_root, MAX_FRAMES)
    #test_dataset  = SeizureDataset(test_csv, data_root, MAX_FRAMES) # Create Test Dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    #test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0) # Create Test Loader
    
    return train_loader, val_loader # Return all 3

def train():
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🚀 Device: {device}")

    # 2. Data Loaders
    train_loader, val_loader = get_dataloaders(SPLIT_FOLDER, DATA_ROOT, BATCH_SIZE)
    print(f"✅ Data Ready: Train ({len(train_loader)} batches), Val ({len(val_loader)} batches)")

    # 3. Initialize Model
    # Note: VSViG_Landmark_Base likely ignores args and uses internal OptInit defaults
    model = VSViG_Landmark_Base() 
    model.to(device)

    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    # 1. Lower LR slightly and increase Weight Decay (L2 Regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)

    # 2. Smarter Scheduler: Reduce LR if Val Loss stops improving for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    epochs = 20
    start_epoch = 0
    min_valid_loss = np.inf
    history = {'train_loss': [], 'val_loss': []}

    # 5. Resume Logic
    if os.path.exists(PATH_TO_LAST_CKPT):
        print(f"🔄 Resuming from checkpoint: {PATH_TO_LAST_CKPT}")
        checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_valid_loss = checkpoint.get('min_valid_loss', np.inf)
        history = checkpoint.get('history', history)

    # --- TRAINING LOOP ---
    for e in range(start_epoch, epochs):
        print(f'\n=== Epoch {e+1}/{epochs} ===')
        
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Unpack: Data, Labels, Mask
            data, labels, mask = batch 
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(data, mask=mask) # Output shape (B, 1)
            
            # Loss Calculation (Ensure shapes match)
            # labels needs to be (B, 1) to match outputs
            loss = criterion(outputs, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'   Train Loss: {avg_train_loss:.4f}')

        # Validation (Every epoch is usually better for small datasets)
        if (e+1) % 1 == 0: 
            model.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    data, labels, mask = batch
                    data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                    
                    outputs = model(data, mask=mask)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    valid_loss += loss.item()
            
            avg_val_loss = valid_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            print(f'   Val Loss:   {avg_val_loss:.4f}')

            # Save Best Model
            if avg_val_loss < min_valid_loss:
                print(f'   🌟 New Best Model! (Loss: {avg_val_loss:.4f})')
                min_valid_loss = avg_val_loss
                torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
        
        # 3. Update Scheduler using VAL LOSS (not step size)
        scheduler.step(avg_val_loss)

        # Save Regular Checkpoint
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, PATH_TO_LAST_CKPT)

        # Save Logs
        with open(PATH_TO_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)      


if __name__ == '__main__':
    train()
