import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent.parent

# --- CONFIGURATION ---
DATA_ROOT =  BASE_DIR /'train_data'
CSV_PATH =  BASE_DIR /'train_data/train_data.csv'
OUTPUT_DIR =  BASE_DIR /'processed_landmarks'
SEED = 42
MIN_CONSECUTIVE_FRAMES = 20  # New threshold

def get_valid_frames_mask(data):
    """Returns a boolean array (N_frames,) where True means the frame has data."""
    # shape: (Frames, 33, 5) -> flatten last two dims -> (Frames, 165)
    flat_frames = data.reshape(data.shape[0], -1)
    # A frame is valid if NOT ALL values are NaN
    return ~np.isnan(flat_frames).all(axis=1)

def is_file_valid(file_path):
    """
    Checks if a file has at least MIN_CONSECUTIVE_FRAMES of valid data.
    """
    if not os.path.exists(file_path):
        return False
        
    try:
        data = np.load(file_path)
        
        # 1. Check Empty/Basic
        if data.size == 0:
            return False
            
        # 2. Get Validity Mask (True = Data, False = NaN)
        valid_mask = get_valid_frames_mask(data)
        
        # 3. Check for Consecutive Sequences
        # We calculate the lengths of consecutive True blocks
        max_consecutive = 0
        current_run = 0
        
        for is_valid in valid_mask:
            if is_valid:
                current_run += 1
            else:
                max_consecutive = max(max_consecutive, current_run)
                current_run = 0
        
        # Final check for the run at the end
        max_consecutive = max(max_consecutive, current_run)
        
        if max_consecutive < MIN_CONSECUTIVE_FRAMES:
            return False
            
        return True
        
    except Exception:
        return False
    
def create_stratified_splits():
    # Generate Folds
    splits_dir = os.path.join(OUTPUT_DIR, 'splits_stratified')
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    # 1. Load Data
    print(f"📂 Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    original_count = len(df)
    
    # 2. FILTERING STAGE
    print(f"🧹 Checking {original_count} files for {MIN_CONSECUTIVE_FRAMES} consecutive valid frames...")
    valid_indices = []
    invalid_files = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['segment_name']
        file_path = os.path.join(DATA_ROOT, filename)
        
        if is_file_valid(file_path):
            valid_indices.append(idx)
        else:
            invalid_files.append(filename)
            
    # Keep only valid rows
    df_clean = df.loc[valid_indices].copy()
    dropped_count = original_count - len(df_clean)
    
    print(f"\n🚫 Removed {dropped_count} invalid files (Empty, All-NaNs, or Non-consecutive).")
    if len(invalid_files) > 0:
        print(f"   Example removed: {invalid_files[0]}")
    print(f"✅ Remaining clean files: {len(df_clean)}")

    # 3. STRATIFICATION logic
    df_clean['child_id'] = df_clean['segment_name'].apply(lambda x: x.split('_')[1])
    
    subject_stats = df_clean.groupby('child_id')['label'].max().reset_index()
    subject_stats.rename(columns={'label': 'has_seizure'}, inplace=True)
    
    seizure_subjects = subject_stats[subject_stats['has_seizure'] == 1]['child_id'].values
    normal_subjects  = subject_stats[subject_stats['has_seizure'] == 0]['child_id'].values
    
    print(f"\n📊 Stratification Groups (After Cleaning):")
    print(f"   Subjects WITH Seizures: {len(seizure_subjects)}")
    print(f"   Subjects WITHOUT Seizures: {len(normal_subjects)}")
    
    def split_ids(id_list, train_ratio=0.70, val_ratio=0.15):
        np.random.seed(SEED)
        np.random.shuffle(id_list)
        
        n_total = len(id_list)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)
        
        train_ids = id_list[:n_train]
        val_ids   = id_list[n_train : n_train + n_val]
        test_ids  = id_list[n_train + n_val:]
        
        return train_ids, val_ids, test_ids

    s_train, s_val, s_test = split_ids(seizure_subjects)
    n_train, n_val, n_test = split_ids(normal_subjects)
    
    final_train_ids = np.concatenate([s_train, n_train])
    final_val_ids   = np.concatenate([s_val, n_val])
    final_test_ids  = np.concatenate([s_test, n_test])
    
    train_df = df_clean[df_clean['child_id'].isin(final_train_ids)].copy()
    val_df   = df_clean[df_clean['child_id'].isin(final_val_ids)].copy()
    test_df  = df_clean[df_clean['child_id'].isin(final_test_ids)].copy()
    
    train_df.to_csv(os.path.join(splits_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test_split.csv'), index=False)

    def print_split_stats(name, dframe, subjects):
        n_clips = len(dframe)
        n_seizure_clips = len(dframe[dframe['label'] == 1])
        pct_seizure = (n_seizure_clips / n_clips * 100) if n_clips > 0 else 0
        subjects_with_seizure = dframe.groupby('child_id')['label'].max().sum()
        
        print(f"\n--- {name} SET ---")
        print(f"   Subjects: {len(subjects)} (Has Seizure: {subjects_with_seizure})")
        print(f"   Total Clips: {n_clips}")
        print(f"   Seizure Clips: {n_seizure_clips} ({pct_seizure:.2f}%)")

    print_split_stats("TRAINING", train_df, final_train_ids)
    print_split_stats("VALIDATION", val_df, final_val_ids)
    print_split_stats("TEST", test_df, final_test_ids)
    
    print(f"\n✅ Cleaned & Stratified splits saved to '{splits_dir}'")

if __name__ == "__main__":
    create_stratified_splits()