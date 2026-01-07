import json
import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
OUTPUT_FOLDER = 'processed_landmarks'
LABEL_CSV = 'train_data.csv' # The original CSV used for ordering

def parse_child_id(filename):
    # Filename format: child_{id}_{segment}.npy
    # Example: child_1_1.npy -> 1
    try:
        parts = filename.split('_')
        # parts[0] = "child", parts[1] = id, parts[2] = segment.npy
        return int(parts[1])
    except:
        return -1

def main():
    if not os.path.exists(LABEL_CSV):
        print(f"❌ Error: {LABEL_CSV} not found.")
        return

    print(f"📂 Loading file list from {LABEL_CSV}...")
    df = pd.read_csv(LABEL_CSV)
    
    # We need to map the dataframe index (which corresponds to the .pt index) to child_id
    # Create a dictionary: child_id -> list of indices
    child_indices = {}
    
    print("🔍 Grouping data by Child ID...")
    for idx, row in df.iterrows():
        filename = row['segment_name']
        label = row['label'] # We keep the label for reference in the json
        
        child_id = parse_child_id(filename)
        
        if child_id == -1:
            print(f"⚠️ Warning: Could not parse child ID from {filename}")
            continue
            
        if child_id not in child_indices:
            child_indices[child_id] = []
            
        # Store [index, label] pair compatible with your dataset class
        child_indices[child_id].append([idx, int(label)])

    unique_children = sorted(child_indices.keys())
    print(f"Found {len(unique_children)} children: {unique_children}")
    
    # Generate Folds
    splits_dir = os.path.join(OUTPUT_FOLDER, 'splits_loocv')
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
        
    print(f"💾 Saving {len(unique_children)} folds to {splits_dir}...")
    
    for val_child_id in unique_children:
        val_data = child_indices[val_child_id]
        train_data = []
        
        # Collect all other children for training
        for train_child_id in unique_children:
            if train_child_id != val_child_id:
                train_data.extend(child_indices[train_child_id])
        
        # Save JSONs
        train_filename = os.path.join(splits_dir, f'train_fold_child_{val_child_id}.json')
        val_filename = os.path.join(splits_dir, f'val_fold_child_{val_child_id}.json')
        
        with open(train_filename, 'w') as f:
            json.dump(train_data, f)
            
        with open(val_filename, 'w') as f:
            json.dump(val_data, f)
            
        print(f"   Created Fold for Child {val_child_id}: Train {len(train_data)} | Val {len(val_data)}")

    print("✅ All splits generated successfully.")

if __name__ == '__main__':
    main()