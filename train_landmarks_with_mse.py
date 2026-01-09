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
        real_idx = self.indices[idx]
        
        # Get data (150, 15, 5)
        # Assuming model expects (15, 150, 5) or similar? 
        # Paper said (Points, Time, Channels) -> (15, 150, 5)
        # We need to permute.
        x = self.all_data[real_idx] # (150, 15, 5)
        x = x.permute(1, 0, 2)      # (15, 150, 5) -> Points, Time, Channels
        
        y = self.all_labels[real_idx].float()
        if self.all_masks is not None:
            mask = self.all_masks[real_idx] # Shape: (150)
            return x, y, mask
        return x, y

    def __len__(self):
        return len(self.indices)

def train(fold_child_id):
    print(f"🚀 Training Fold: Leave-One-Out Child {fold_child_id}")
    
    # 1. Load Data Into Memory (Once)
    print("   Loading tensors...")
    all_data = torch.load(LANDMARKS_FILE)
    all_labels = torch.load(LABELS_FILE)
    all_masks = torch.load(MASKS_FILE) # Use if model supports masking
    
    # 2. Setup Splits
    split_dir = os.path.join(DATA_FOLDER, 'splits_loocv')
    train_idx_path = os.path.join(split_dir, f'train_indices_child_{fold_child_id}.json')
    val_idx_path = os.path.join(split_dir, f'val_indices_child_{fold_child_id}.json')
    
    if not os.path.exists(train_idx_path):
        print(f"❌ Split file not found: {train_idx_path}")
        return

    # 3. Create Datasets
    train_ds = LandmarkDataset(train_idx_path, all_data, all_labels, all_masks)
    val_ds = LandmarkDataset(val_idx_path, all_data, all_labels, all_masks)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 4. Initialize Model (Needs adaptation for Landmarks!)
    # We need to modify VSViG.py to accept 5 channels instead of image patches
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    model = VSViG_Landmark_Base(num_joints=15, in_channels=5, num_classes=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    #MSE = nn.MSELoss()
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


    # --- RESUME LOGIC STARTS HERE ---
    start_epoch = 0 #change manually if needed
    min_valid_loss = np.inf
    #history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    history = {'train_loss': [], 'val_loss': []}

    # Check for 'last_checkpoint.pth' (Full state for resuming)
    if os.path.exists(PATH_TO_LAST_CKPT):
        print(f"Found checkpoint: {PATH_TO_LAST_CKPT}. Resuming...")
        checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_valid_loss = checkpoint['min_valid_loss']
        
        # Load history if it exists
        if 'history' in checkpoint:
            history = checkpoint['history']

    # Fallback: If only 'best_model.pth' exists
    elif os.path.exists(PATH_TO_BEST_MODEL):
        print(f"Found best_model.pth but no checkpoint. Loading weights only.")
        # We assume a weight-only save for the old file
        try:
            model.load_state_dict(torch.load(PATH_TO_BEST_MODEL, map_location=device))
        except:
            print("Could not load best_model.pth weights. Starting fresh.")
    # --- RESUME LOGIC ENDS HERE ---
    train_loss_stack = []
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        print(f'===================================\n Running Epoch: {e+1} \n===================================')

# Inside training loop
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                data, labels, mask = batch
                data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                outputs = model(data, mask=mask) # Pass mask to model
            else:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
            # print(outputs)
            #loss = MSE(outputs.squeeze().float(),labels.float())
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_stack.append(loss.item())
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Training Loss: {train_loss:.3f}')

        if (e+1)%5 == 0:
            valid_loss = 0.0
            #RMSE_loss = 0.0
            _iter = 0
            model.eval()
            with torch.no_grad():
                for sample, labels in val_loader:
                    data = sample.to(device)
                    labels = labels.to(device)
                    outputs = model(data)
                    #loss = MSE(outputs.squeeze(),labels)
                    loss = criterion(outputs.squeeze(), labels)
                    valid_loss += loss.item()
                    #RMSE_loss += torch.sqrt(MSE(outputs,labels)).item()*100
                    _iter += 1
            avg_val_loss = valid_loss / len(val_loader)
            #avg_rmse = RMSE_loss / len(val_loader)
            
            history['val_loss'].append(avg_val_loss)
            #history['val_rmse'].append(avg_rmse)
            print(f' +++++++++++++++++++++++++++++++++++\n Val Loss: {valid_loss:.3f} \n +++++++++++++++++++++++++++++++++++')

            if min_valid_loss > valid_loss:
                print(f'   -> Saving new best model to {PATH_TO_BEST_MODEL}')
                min_valid_loss = valid_loss
                torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
        scheduler.step()
        # 5. SAVE REGULAR CHECKPOINT (Every Epoch)
        # This allows you to resume even if you crash between validation runs
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, PATH_TO_LAST_CKPT)

        # 6. SAVE LOGS TO FILE
        with open(PATH_TO_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child_id', type=int, required=True, help='Child ID to hold out for validation')
    args = parser.parse_args()
    
    train(args.child_id)