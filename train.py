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
        record = self.indices[idx] 
        real_idx = record[0]
        
        # Get data
        x = self.all_data[real_idx] 
        
        # --- FIX: STRICT SHAPE ENFORCEMENT ---
        # Ensure x is exactly (150, 15, 5)
        # This handles cases where x might be (1, 150, 15, 5) or (150, 15, 5, 1)
        # Reshape is safe because the element count (150*15*5) is constant.
        if x.shape != (150, 15, 5):
            x = x.reshape(150, 15, 5)
            
        y = self.all_labels[real_idx].float()
        
        if self.all_masks is not None:
            mask = self.all_masks[real_idx] 
            # Ensure mask is exactly (150)
            if mask.shape != (150,):
                mask = mask.reshape(150)
            return x, y, mask
            
        return x, y

    def __len__(self):
        return len(self.indices)
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
def train(args):
    # Setup Paths based on Child ID
    child_id = args.child_id
    print(f"🚀 Training Fold: Child {child_id} | LR: {args.lr} | Batch: {args.batch_size}")
    
    # Specific save paths for this fold/experiment
    save_path = os.path.join(CHECKPOINT_DIR, f"best_model_child_{child_id}.pth")
    log_path = os.path.join(CHECKPOINT_DIR, f"log_child_{child_id}.json")

    # 1. Load Data
    print("   Loading tensors...")
    all_data = torch.load(LANDMARKS_FILE)
    all_labels = torch.load(LABELS_FILE)
    all_masks = torch.load(MASKS_FILE)
    
    split_dir = os.path.join(DATA_FOLDER, 'splits_loocv')
    train_idx_path = os.path.join(split_dir, f'train_indices_child_{child_id}.json')
    val_idx_path = os.path.join(split_dir, f'val_indices_child_{child_id}.json')
    
    if not os.path.exists(train_idx_path):
        print(f"❌ Split file not found: {train_idx_path}")
        return

    # 2. Datasets & Loaders
    train_ds = LandmarkDataset(train_idx_path, all_data, all_labels, all_masks)
    val_ds = LandmarkDataset(val_idx_path, all_data, all_labels, all_masks)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. Model
    # Note: Ensure VSViG_Landmark_Base uses args.dropout if you pass it
    model = VSViG_Landmark_Base() 
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    
    # 4. Optimizer & Loss
    # Use BCEWithLogitsLoss for binary classification stability
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # Adjusted for lower epochs
    
    early_stopper = EarlyStopping(patience=args.patience)
    
    history = {'train_loss': [], 'val_loss': []}
    min_valid_loss = np.inf

    # 5. Training Loop
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:
                data, labels, mask = batch
                data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                outputs = model(data, mask=mask)
            else:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
            
            # Important: If model has Sigmoid at end, remove it for BCEWithLogitsLoss
            # Or use BCELoss if model keeps Sigmoid.
            # Assuming model outputs probabilities (Sigmoid):
            # loss = criterion(outputs.squeeze(), labels) -> This requires BCELoss
            # If using BCEWithLogitsLoss, model must output raw logits (remove torch.sigmoid in forward)
            
            # Since we didn't change model code here, let's stick to BCELoss if model has sigmoid
            # OR wrap outputs in logit() to invert sigmoid before BCEWithLogitsLoss
            # Safe bet: Use BCELoss since model has Sigmoid.
            
            loss_fn = nn.BCEWithLogitsLoss() 
            loss = loss_fn(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    data, labels, mask = batch
                    data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                    outputs = model(data, mask=mask)
                else:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                
                loss = loss_fn(outputs.squeeze(), labels)
                valid_loss += loss.item()
        
        avg_val_loss = valid_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {e+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < min_valid_loss:
            min_valid_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            # print("  Saved Best Model")
            
        scheduler.step()
        
        # Early Stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping triggered at epoch {e+1}")
            break

    # Save Logs
    with open(log_path, 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child_id', type=int, required=True, help='Child ID for LOOCV')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (needs model support)')
    
    args = parser.parse_args()
    train(args)