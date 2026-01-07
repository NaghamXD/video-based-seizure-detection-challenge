import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
CSV_PATH = 'train_data/train_data.csv'   # Path to your existing labels
OUTPUT_DIR = 'processed_landmarks'    # Output folder
SEED = 42                                # Random seed for reproducibility

def create_stratified_splits():
    # Generate Folds
    splits_dir = os.path.join(OUTPUT_DIR, 'splits_stratified')
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Extract Child ID (assuming format 'child_100_1.npy')
    # This splits by '_' and takes the middle number
    df['child_id'] = df['segment_name'].apply(lambda x: x.split('_')[1])
    
    # 2. Analyze Each Child
    # We group by child_id to see if they EVER had a seizure
    # 'max' of label works: if they have ANY 1s, max will be 1. If all 0s, max is 0.
    subject_stats = df.groupby('child_id')['label'].max().reset_index()
    subject_stats.rename(columns={'label': 'has_seizure'}, inplace=True)
    
    # Separate into two lists
    seizure_subjects = subject_stats[subject_stats['has_seizure'] == 1]['child_id'].values
    normal_subjects  = subject_stats[subject_stats['has_seizure'] == 0]['child_id'].values
    
    print(f"📊 Dataset Analysis:")
    print(f"   Total Subjects: {len(subject_stats)}")
    print(f"   Subjects WITH Seizures: {len(seizure_subjects)}")
    print(f"   Subjects WITHOUT Seizures: {len(normal_subjects)}")
    
    # 3. Helper Function to Split a List of IDs
    def split_ids(id_list, train_ratio=0.70, val_ratio=0.15):
        np.random.seed(SEED)
        np.random.shuffle(id_list)
        
        n_total = len(id_list)
        n_train = int(n_total * train_ratio) #run per id - not per clip
        n_val   = int(n_total * val_ratio)
        
        train_ids = id_list[:n_train]
        val_ids   = id_list[n_train : n_train + n_val]
        test_ids  = id_list[n_train + n_val:]
        
        return train_ids, val_ids, test_ids

    # 4. Split BOTH groups separately
    # Split the "Seizure Kids"
    s_train, s_val, s_test = split_ids(seizure_subjects)
    # Split the "Normal Kids"
    n_train, n_val, n_test = split_ids(normal_subjects)
    
    # 5. Combine them
    final_train_ids = np.concatenate([s_train, n_train])
    final_val_ids   = np.concatenate([s_val, n_val])
    final_test_ids  = np.concatenate([s_test, n_test])
    
    # 6. Create the final DataFrames
    train_df = df[df['child_id'].isin(final_train_ids)].copy()
    val_df   = df[df['child_id'].isin(final_val_ids)].copy()
    test_df  = df[df['child_id'].isin(final_test_ids)].copy()
    
    # 7. Save to CSV
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_split.csv'), index=False)

    # 8. Detailed Report
    def print_split_stats(name, dframe, subjects):
        n_clips = len(dframe)
        n_seizure_clips = len(dframe[dframe['label'] == 1])
        pct_seizure = (n_seizure_clips / n_clips * 100) if n_clips > 0 else 0
        
        # Check how many subjects in this split actually have a seizure file
        # (This is the critical check!)
        subjects_with_seizure = dframe.groupby('child_id')['label'].max().sum()
        
        print(f"\n--- {name} SET ---")
        print(f"   Subjects: {len(subjects)} (Has Seizure: {subjects_with_seizure})")
        print(f"   Total Clips: {n_clips}")
        print(f"   Seizure Clips: {n_seizure_clips} ({pct_seizure:.2f}%)")

    print_split_stats("TRAINING", train_df, final_train_ids)
    print_split_stats("VALIDATION", val_df, final_val_ids)
    print_split_stats("TEST", test_df, final_test_ids)
    
    print(f"\n✅ Stratified splits saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    create_stratified_splits()