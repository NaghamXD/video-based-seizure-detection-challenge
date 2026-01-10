import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent dir to path for model import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from VSViG_Landmark import VSViG_Landmark_Base

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / 'train_data'
SPLIT_FOLDER = BASE_DIR / 'processed_landmarks/splits_stratified'
CHECKPOINT_DIR = BASE_DIR / "checkpoints_curriculum"

# Training Settings
MAX_FRAMES = 150
BATCH_SIZE = 32
TOTAL_EPOCHS = 100
PHASE_2_START_EPOCH = 15  # Switch to full data after 15 epochs
CLEAN_THRESHOLD = 140     # Clips with >= 140 frames are considered "Perfect"

# Mapping (MediaPipe -> VSViG)
MP_MAP = {
    0: [0], 1: [2], 2: [5], 3: [12], 4: [14], 5: [16], 
    6: [24], 7: [26], 8: [28], 9: [11], 10: [13], 11: [15], 
    12: [23], 13: [25], 14: [27]
}

if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model_curriculum.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

# --- DATASET CLASS (With Left-Align Logic) ---
class SeizureDataset(Dataset):
    def __init__(self, df, root_dir, max_frames=150):
        self.annotations = df
        self.root_dir = root_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        filename = row['segment_name']
        label = int(row['label'])
        file_path = os.path.join(self.root_dir, filename)

        try:
            raw_lmks = np.load(file_path)
            if raw_lmks.size == 0: raw_lmks = np.zeros((1, 33, 5), dtype=np.float32)
        except:
            raw_lmks = np.zeros((1, 33, 5), dtype=np.float32)

        # Trim Leading NaNs (Left-Align)
        if raw_lmks.shape[0] > 0:
            flat_frames = raw_lmks.reshape(raw_lmks.shape[0], -1)
            valid_mask = ~np.isnan(flat_frames).all(axis=1)
            if valid_mask.any():
                first_valid_idx = np.argmax(valid_mask)
                raw_lmks = raw_lmks[first_valid_idx:]
            else:
                raw_lmks = np.zeros((0, 33, 5))

        mapped_data = np.zeros((self.max_frames, 15, 5), dtype=np.float32)
        mask = np.zeros((self.max_frames), dtype=np.float32)

        num_frames = min(self.max_frames, raw_lmks.shape[0])
        
        if num_frames > 0:
            current_raw = raw_lmks[:num_frames].copy()
            
            # Normalize
            current_raw[:, :, 0] -= 0.5
            current_raw[:, :, 1] -= 0.5
            current_raw[:, :, 0:3] /= 6.0

            # Map
            for target_idx, mp_indices in MP_MAP.items():
                mapped_data[:num_frames, target_idx, :] = current_raw[:, mp_indices[0], :]

            # Mask (1=Data, 0=Pad)
            flat_slice = current_raw.reshape(num_frames, -1)
            mask[:num_frames] = (~np.isnan(flat_slice).all(axis=1)).astype(np.float32)
            mapped_data = np.nan_to_num(mapped_data, nan=0.0)

        return torch.from_numpy(mapped_data), torch.tensor(label, dtype=torch.float32), torch.from_numpy(mask)

# --- HELPER: FILTER DATA ---
def get_clean_dataframe(full_df, root_dir, threshold=140):
    """
    Scans files and returns a DataFrame containing only clips 
    with >= threshold valid frames.
    """
    print(f"🧹 Scanning {len(full_df)} files to find 'Perfect' data (>={threshold} frames)...")
    valid_indices = []
    
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        file_path = os.path.join(root_dir, row['segment_name'])
        try:
            if not os.path.exists(file_path): continue
            
            # Quick check without loading full array if possible, 
            # but usually we must load to check NaNs
            data = np.load(file_path)
            
            # Calculate valid frames length
            flat = data.reshape(data.shape[0], -1)
            valid_count = (~np.isnan(flat).all(axis=1)).sum()
            
            if valid_count >= threshold:
                valid_indices.append(idx)
        except:
            continue
            
    clean_df = full_df.loc[valid_indices].copy()
    print(f"✅ Found {len(clean_df)} 'Clean' clips (Phase 1 Data).")
    return clean_df

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. Load CSVs
    full_train_df = pd.read_csv(os.path.join(SPLIT_FOLDER, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(SPLIT_FOLDER, 'val_split.csv'))

    # 2. Prepare Phase 1 Data (Clean Only)
    clean_train_df = get_clean_dataframe(full_train_df, DATA_ROOT, CLEAN_THRESHOLD)
    
    # 3. Create Datasets
    # Phase 1 Dataset
    ds_phase1 = SeizureDataset(clean_train_df, DATA_ROOT, MAX_FRAMES)
    dl_phase1 = DataLoader(ds_phase1, batch_size=BATCH_SIZE, shuffle=True)
    
    # Phase 2 Dataset (Full)
    ds_phase2 = SeizureDataset(full_train_df, DATA_ROOT, MAX_FRAMES)
    dl_phase2 = DataLoader(ds_phase2, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation (Always use Full Validation set to track real progress)
    ds_val = SeizureDataset(val_df, DATA_ROOT, MAX_FRAMES)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Setup
    model = VSViG_Landmark_Base()
    model.to(device)
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = 0.3

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. Training State
    min_valid_loss = np.inf
    history = {'train_loss': [], 'val_loss': [], 'phase': []}
    start_epoch = 0
    
    # Resume Logic
    if os.path.exists(PATH_TO_LAST_CKPT):
        print(f"🔄 Resuming...")
        ckpt = torch.load(PATH_TO_LAST_CKPT, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        min_valid_loss = ckpt['min_valid_loss']
        history = ckpt.get('history', history)

    # --- MAIN LOOP ---
    for e in range(start_epoch, TOTAL_EPOCHS):
        
        # --- CURRICULUM SWITCH LOGIC ---
        if e < PHASE_2_START_EPOCH:
            current_loader = dl_phase1
            phase_name = "PHASE 1 (Clean Data)"
        else:
            current_loader = dl_phase2
            phase_name = "PHASE 2 (Full Data)"
            
        print(f'\n=== Epoch {e+1}/{TOTAL_EPOCHS} | {phase_name} ===')
        
        # Train
        model.train()
        train_loss = 0.0
        for batch in current_loader:
            data, labels, mask = batch 
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, mask=mask)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(current_loader)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in dl_val:
                data, labels, mask = batch
                data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                outputs = model(data, mask=mask)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()
        
        avg_val_loss = valid_loss / len(dl_val)
        
        # Logging
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['phase'].append(1 if e < PHASE_2_START_EPOCH else 2)
        
        print(f'   Train Loss: {avg_train_loss:.4f}')
        print(f'   Val Loss:   {avg_val_loss:.4f}')

        # Save Best
        if avg_val_loss < min_valid_loss:
            print(f'   🌟 New Best Model! (Loss: {avg_val_loss:.4f})')
            min_valid_loss = avg_val_loss
            torch.save(model.state_dict(), PATH_TO_BEST_MODEL)

        scheduler.step(avg_val_loss)
        
        # Save Checkpoint
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, PATH_TO_LAST_CKPT)

        with open(PATH_TO_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == '__main__':
    train()