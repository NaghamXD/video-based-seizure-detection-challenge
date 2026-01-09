import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from VSViG_Landmark import VSViG_Landmark_Base
from t_lm import LandmarkDataset, CHECKPOINT_DIR # Re-use dataset class

# --- CONFIGURATION ---
DATA_FOLDER = 'processed_landmarks'
LANDMARKS_FILE = os.path.join(DATA_FOLDER, 'all_landmarks.pt')
LABELS_FILE = os.path.join(DATA_FOLDER, 'all_labels.pt')
MASKS_FILE = os.path.join(DATA_FOLDER, 'all_masks.pt')

# Target fold for tuning (Pick a child with seizures!)
TARGET_CHILD_ID = 102 
MAX_EPOCHS = 30
PATIENCE = 8 # Early stopping patience

# Hyperparameter Grid
GRID = {
    'lr': [1e-4, 1e-5],
    'dropout': [0.3, 0.5],
    'batch_size': [16, 32]
}

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

def train_config(config, train_loader, val_loader, device):
    print(f"   Testing Config: {config}")
    
    # Initialize Model with specific dropout (requires modifying model init if not passed)
    # Assuming VSViG_Landmark_Base can take dropout as kwarg or we modify it manually
    # Ideally, update VSViG_Landmark.py to accept dropout in OptInit
    model = VSViG_Landmark_Base() 
    
    # Manually update dropout layers if model doesn't support init arg yet
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = config['dropout']
            
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    early_stopper = EarlyStopping(patience=PATIENCE)
    history = {'train_loss': [], 'val_loss': []}
    best_loss_in_run = np.inf
    for epoch in range(MAX_EPOCHS):
        # TRAIN
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
            
            # If model has Sigmoid, use BCELoss. If raw logits, use BCEWithLogitsLoss.
            # Assuming model has Sigmoid from previous discussions -> BCELoss
            # BUT efficient tuning prefers logits. Let's stick to what worked:
            # If model output is [0,1], use BCELoss.
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATE
        model.eval()
        val_loss = 0.0
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
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # CHANGE 2: SAVE CHECKPOINT
        if avg_val_loss < best_loss_in_run:
            best_loss_in_run = avg_val_loss
            # Create a unique name for this config
            save_name = f"model_lr{config['lr']}_bs{config['batch_size']}_drop{config['dropout']}.pth"
            save_path = os.path.join(CHECKPOINT_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            # print(f"        💾 Saved best model to {save_name}")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"     Ep {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"     🛑 Early stopping at epoch {epoch+1}")
            break
            
    return history, early_stopper.best_loss

def main():
    # 1. Load Data Once
    print("📂 Loading Data...")
    all_data = torch.load(LANDMARKS_FILE)
    all_labels = torch.load(LABELS_FILE)
    all_masks = torch.load(MASKS_FILE)
    
    # 2. Setup Splits for Target Child
    split_dir = os.path.join(DATA_FOLDER, 'splits_loocv')
    train_idx = os.path.join(split_dir, f'train_fold_child_{TARGET_CHILD_ID}.json')
    val_idx = os.path.join(split_dir, f'val_fold_child_{TARGET_CHILD_ID}.json')
    
    train_ds = LandmarkDataset(train_idx, all_data, all_labels, all_masks)
    val_ds = LandmarkDataset(val_idx, all_data, all_labels, all_masks)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 3. Grid Search
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = {}
    
    print(f"🔎 Starting Grid Search on {len(combinations)} configurations...")
    
    for i, config in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)} ---")
        
        # Reload loaders if batch size changes
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        history, best_loss = train_config(config, train_loader, val_loader, device)
        
        config_str = f"LR{config['lr']}_BS{config['batch_size']}_Drop{config['dropout']}"
        results[config_str] = {
            'best_val_loss': best_loss,
            'history': history
        }

    # 4. Visualization & Reporting
    print("\n🏆 Results Summary:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])
    
    for name, data in sorted_results:
        print(f"  {name}: {data['best_val_loss']:.4f}")
        
    # Plotting
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        # Plot only validation loss for clarity
        plt.plot(data['history']['val_loss'], label=f"{name} (Val)")
        
    plt.title(f'Hyperparameter Tuning (Child {TARGET_CHILD_ID})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'tuning_results.png'))
    print(f"📉 Plot saved to {CHECKPOINT_DIR}/tuning_results.png")

if __name__ == '__main__':
    main()