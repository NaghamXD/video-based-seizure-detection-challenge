import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import argparse
from VSViG_Landmark import *

# --- CONFIGURATION ---
DATA_FOLDER = 'processed_landmarks'
LANDMARKS_FILE = os.path.join(DATA_FOLDER, 'all_landmarks.pt')
LABELS_FILE = os.path.join(DATA_FOLDER, 'all_labels.pt')
MASKS_FILE = os.path.join(DATA_FOLDER, 'all_masks.pt') # Optional usage

CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

class LandmarkDataset(Dataset):
    def __init__(self, indices_file, all_data, all_labels, all_masks=None):
        """
        indices_file: Path to JSON containing list of indices for this split
        all_data: The big tensor (N, 150, 15, 5) loaded in memory
        all_labels: The big tensor (N) loaded in memory
        """
        with open(indices_file, 'r') as f:
            self.indices = json.load(f)
            
        self.all_data = all_data
        self.all_labels = all_labels
        self.all_masks = all_masks

    def __getitem__(self, idx):
        # ROOT CAUSE FIX: The JSON contains [index, label]. 
        # We only want the index (the 0-th element).
        record = self.indices[idx] 
        real_idx = record[0] # <--- THIS IS THE KEY FIX
        
        # Now real_idx is just an integer (e.g., 105), not a list [105, 1]
        
        # 1. Get Data (No shape hacks needed anymore!)
        x = self.all_data[real_idx]
        
        # 2. Reshape if needed (Standard safety check)
        if x.dim() == 4 and x.shape[0] == 1:
             x = x.squeeze(0)
        
        if x.shape != (150, 15, 5):
             x = x.reshape(150, 15, 5)

        # 3. Get Label
        y = self.all_labels[real_idx].float()
        
        # 4. Get Mask
        if self.all_masks is not None:
            mask = self.all_masks[real_idx]
            if mask.shape != (150,):
                mask = mask.reshape(150)
            return x, y, mask
            
        return x, y

    def __len__(self):
        return len(self.indices)

def train(fold_child_id):
    # This is a template file used by imports. 
    # Logic is mainly in the Dataset xclass above.
    pass