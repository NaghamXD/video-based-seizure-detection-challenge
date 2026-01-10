import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from VSViG_Landmark import VSViG_Landmark_Base

# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent.parent

# --- CONFIGURATION ---
DATA_ROOT =  BASE_DIR /'train_data'            
SPLIT_FOLDER =  BASE_DIR /'processed_landmarks/splits_stratified' 
MAX_FRAMES = 150                    
BATCH_SIZE = 32
CHECKPOINT_DIR = BASE_DIR / "checkpoints_stratified"

# Mapping 33 MediaPipe Landmarks to 15 VSViG Joints
MP_MAP = {
    0: [0], 1: [2], 2: [5], 3: [12], 4: [14], 5: [16], 
    6: [24], 7: [26], 8: [28], 9: [11], 10: [13], 11: [15], 
    12: [23], 13: [25], 14: [27]
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

        # 3. Data Processing & Mapping
        mapped_data = np.zeros((self.max_frames, 15, 5), dtype=np.float32)
        mask = np.zeros((self.max_frames), dtype=np.float32) 

        # --- TRIM LEADING NANS ---
        # Find where the data actually starts
        if raw_lmks.shape[0] > 0:
            # Flatten to check frame validity
            flat_frames = raw_lmks.reshape(raw_lmks.shape[0], -1)
            valid_mask = ~np.isnan(flat_frames).all(axis=1) # Boolean array
            
            if valid_mask.any():
                # Find index of first True
                first_valid_idx = np.argmax(valid_mask)
                # Slice the array to start from valid data
                current_raw = raw_lmks[first_valid_idx:]
            else:
                # File is all NaNs
                current_raw = np.zeros((0, 33, 5))
        else:
            current_raw = raw_lmks

        # Take up to MAX_FRAMES of the *trimmed* data
        num_frames = min(self.max_frames, current_raw.shape[0])
        
        if num_frames > 0:
            current_raw = current_raw[:num_frames].copy()

            # --- NORMALIZATION ---
            current_raw[:, :, 0] = current_raw[:, :, 0] - 0.5
            current_raw[:, :, 1] = current_raw[:, :, 1] - 0.5
            current_raw[:, :, 0:3] = current_raw[:, :, 0:3] / 6.0

            # --- MAPPING 33 -> 15 ---
            for target_idx, mp_indices in MP_MAP.items():
                source_idx = mp_indices[0]
                mapped_data[:num_frames, target_idx, :] = current_raw[:, source_idx, :]

            # --- SMART MASKING ---
            # Mask is 1 ONLY if the frame has data (after trimming).
            # If there are internal NaNs in the trimmed sequence, we want mask=0 for those specific frames.
            # We re-check validity on the *mapped* or *trimmed* raw data.
            
            # Re-calculate valid mask for the slice we actually kept
            flat_slice = current_raw.reshape(num_frames, -1)
            slice_validity = ~np.isnan(flat_slice).all(axis=1)
            
            # Fill mask
            mask[:num_frames] = slice_validity.astype(np.float32)
            
            # Handle remaining NaNs in data (turn to 0.0)
            mapped_data = np.nan_to_num(mapped_data, nan=0.0)

        # 4. Convert to Tensor
        tensor_data = torch.from_numpy(mapped_data)
        tensor_label = torch.tensor(label, dtype=torch.float32)
        tensor_mask = torch.from_numpy(mask) 

        return tensor_data, tensor_label, tensor_mask

def get_dataloaders(split_dir, data_root, batch_size=32):
    train_csv = os.path.join(split_dir, 'train_split.csv')
    val_csv   = os.path.join(split_dir, 'val_split.csv')
    
    train_dataset = SeizureDataset(train_csv, data_root, MAX_FRAMES)
    val_dataset   = SeizureDataset(val_csv, data_root, MAX_FRAMES)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train():
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"🚀 Device: {device}")

    train_loader, val_loader = get_dataloaders(SPLIT_FOLDER, DATA_ROOT, BATCH_SIZE)
    print(f"✅ Data Ready: Train ({len(train_loader)} batches), Val ({len(val_loader)} batches)")

    model = VSViG_Landmark_Base() 
    model.to(device)
    
    # Inject Dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.3

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    epochs = 200
    start_epoch = 0
    min_valid_loss = np.inf
    history = {'train_loss': [], 'val_loss': []}

    early_stopping_patience = 50 
    early_stopping_counter = 0

    if os.path.exists(PATH_TO_LAST_CKPT):
        print(f"🔄 Resuming from checkpoint: {PATH_TO_LAST_CKPT}")
        checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_valid_loss = checkpoint.get('min_valid_loss', np.inf)
        history = checkpoint.get('history', history)

    for e in range(start_epoch, epochs):
        print(f'\n=== Epoch {e+1}/{epochs} ===')
        
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            data, labels, mask = batch 
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, mask=mask) 
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'   Train Loss: {avg_train_loss:.4f}')

        # Validation
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
        
        if avg_val_loss < min_valid_loss:
            print(f'   🌟 New Best Model! (Loss: {avg_val_loss:.4f})')
            min_valid_loss = avg_val_loss
            torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
            early_stopping_counter = 0 
        else:
            early_stopping_counter += 1
            print(f'   ⚠️ No improvement for {early_stopping_counter}/{early_stopping_patience} epochs.')
            
        if early_stopping_counter >= early_stopping_patience:
            print(f'\n🛑 Early Stopping Triggered! Val loss has not improved for {early_stopping_patience} epochs.')
            break

        scheduler.step(avg_val_loss)

        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, PATH_TO_LAST_CKPT)

        with open(PATH_TO_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)      

if __name__ == '__main__':
    train()