import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / 'processed_landmarks/splits_stratified'

def analyze_set(name, csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ {name}: File not found at {csv_path}")
        return None, set()

    df = pd.read_csv(csv_path)
    
    # 1. Basic Counts
    total = len(df)
    seizures = df[df['label'] == 1].shape[0]
    normals = df[df['label'] == 0].shape[0]
    ratio = (seizures / total * 100) if total > 0 else 0
    
    # 2. Patient IDs
    # Assuming filename format: "segment_child_ID_index.npy" or similar
    # Adjust the split logic if your naming convention is different
    # Based on your previous code: "child_id" column might not exist in the csv, 
    # so we re-extract it from segment_name
    child_ids = df['segment_name'].apply(lambda x: x.split('_')[1]).unique()
    
    print(f"📊 {name} SET Analysis:")
    print(f"   - Total Clips:    {total}")
    print(f"   - Seizure Clips:  {seizures} ({ratio:.2f}%)")
    print(f"   - Normal Clips:   {normals}")
    print(f"   - Unique Patients:{len(child_ids)}")
    print("-" * 30)
    
    return df, set(child_ids)

def verify():
    print(f"🔍 Verifying splits in: {SPLIT_DIR}\n")
    
    train_df, train_patients = analyze_set("TRAIN", SPLIT_DIR / 'train_split.csv')
    val_df, val_patients = analyze_set("VAL",   SPLIT_DIR / 'val_split.csv')
    test_df, test_patients = analyze_set("TEST",  SPLIT_DIR / 'test_split.csv')
    
    print("\n🕵️‍♀️ CHECKING FOR DATA LEAKAGE (Patient Overlap)...")
    
    # Check Intersections
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    has_leakage = False
    
    if train_val_overlap:
        print(f"❌ DANGER: Leakage between TRAIN and VAL! Patients: {train_val_overlap}")
        has_leakage = True
    else:
        print("✅ Train vs Val: No overlap.")

    if train_test_overlap:
        print(f"❌ DANGER: Leakage between TRAIN and TEST! Patients: {train_test_overlap}")
        has_leakage = True
    else:
        print("✅ Train vs Test: No overlap.")
        
    if val_test_overlap:
        print(f"❌ DANGER: Leakage between VAL and TEST! Patients: {val_test_overlap}")
        has_leakage = True
    else:
        print("✅ Val vs Test:   No overlap.")
        
    if not has_leakage:
        print("\n🎉 SUCCESS: All sets are fully independent by patient.")
    else:
        print("\n🛑 FAILURE: You have data leakage. Do not train.")

if __name__ == "__main__":
    verify()