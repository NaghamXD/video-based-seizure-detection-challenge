import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
import itertools
from VSViG_Landmark import VSViG_Landmark_Base
from stratified_data.train_stratified import SeizureDataset, get_dataloaders # Re-use your existing code

from pathlib import Path

# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent
# ↑ adjust number of `.parent` depending on depth

# --- CONFIGURATION ---
DATA_ROOT = BASE_DIR /'train_data'
SPLIT_FOLDER = BASE_DIR /'processed_landmarks/splits_stratified'
OUTPUT_LOG = BASE_DIR /'hp_tuning_stratified_results.json'

# 1. UPDATED GRID (Based on Overfitting Diagnosis)
HP_GRID = {
    'learning_rate': [1e-4, 5e-5],      # Lower rates to fix volatility
    'dropout':       [0.3, 0.5],        # Standard regularization
    'batch_size':    [32],              # Keep fixed to save time, or add [16]
    'weight_decay':  [1e-4, 1e-3]       # CRITICAL: Stronger L2 regularization
}

# 2. FIXED PARAMETERS
EPOCHS_PER_RUN = 8  # Increased to 8 so Scheduler has time to work
MAX_FRAMES = 150

def train_one_config(config, device):
    print(f"   Testing Config: {config}")
    
    # A. Setup Data
    train_loader, val_loader = get_dataloaders(SPLIT_FOLDER, DATA_ROOT, batch_size=config['batch_size'])
    
    # B. Setup Model
    model = VSViG_Landmark_Base() 
    model.to(device)
    
    # Inject Dropout Value (Hack for VSViG)
    # This works assuming you fixed the .forward() method in VSViG_Landmark.py!
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = config['dropout']
    
    # C. Setup Optimizer & Scheduler
    criterion = nn.BCEWithLogitsLoss()
    
    # Use the config's Weight Decay and LR
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Smart Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=False
    )
    
    # D. Training Loop
    best_val_loss = float('inf')
    
    for e in range(EPOCHS_PER_RUN):
        # Train
        model.train()
        for batch in train_loader:
            data, labels, mask = batch
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, mask=mask)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, labels, mask = batch
                data, labels, mask = data.to(device), labels.to(device), mask.to(device)
                
                outputs = model(data, mask=mask)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update Best Score
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
        # Step the Scheduler
        scheduler.step(avg_val_loss)
            
    print(f"      -> Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"🚀 Starting Grid Search on {device}...")
    
    # Generate combinations
    keys, values = zip(*HP_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    for i, config in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)} ---")
        try:
            loss = train_one_config(config, device)
            
            result_entry = config.copy()
            result_entry['best_val_loss'] = loss
            results.append(result_entry)
            
        except Exception as e:
            print(f"❌ Run failed: {e}")
            
    # Sort by lowest loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    print("\n🏆 TOP 3 CONFIGURATIONS:")
    for i in range(min(3, len(results))):
        print(f"   {i+1}. Loss: {results[i]['best_val_loss']:.4f} | {results[i]}")
        
    with open(OUTPUT_LOG, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Full results saved to {OUTPUT_LOG}")

if __name__ == '__main__':
    main()