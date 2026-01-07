import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = 'train_data'            # Folder containing .npy files
LABEL_CSV = 'train_data/train_data.csv'      # Path to the labels CSV

# MediaPipe Pose Landmark Names (Index 0-32)
LANDMARK_NAMES = {
    0: 'Nose', 1: 'Left Eye Inner', 2: 'Left Eye', 3: 'Left Eye Outer', 
    4: 'Right Eye Inner', 5: 'Right Eye', 6: 'Right Eye Outer', 
    7: 'Left Ear', 8: 'Right Ear', 9: 'Mouth Left', 10: 'Mouth Right',
    11: 'Left Shoulder', 12: 'Right Shoulder', 
    13: 'Left Elbow', 14: 'Right Elbow', 
    15: 'Left Wrist', 16: 'Right Wrist', 
    17: 'Left Pinky', 18: 'Right Pinky', 
    19: 'Left Index', 20: 'Right Index', 
    21: 'Left Thumb', 22: 'Right Thumb', 
    23: 'Left Hip', 24: 'Right Hip', 
    25: 'Left Knee', 26: 'Right Knee', 
    27: 'Left Ankle', 28: 'Right Ankle', 
    29: 'Left Heel', 30: 'Right Heel', 
    31: 'Left Foot Index', 32: 'Right Foot Index'
}

def analyze_out_of_bounds():
    # 1. Load Labels
    if not os.path.exists(LABEL_CSV):
        print(f"❌ Error: CSV not found at {LABEL_CSV}")
        return

    df = pd.read_csv(LABEL_CSV)
    print(f"📂 Found {len(df)} files in CSV.")

    # 2. Initialize Counters (Shape: [33 landmarks])
    # Structure: [Count_Normal, Count_Seizure]
    total_frames = {0: np.zeros(33), 1: np.zeros(33)}
    oob_frames   = {0: np.zeros(33), 1: np.zeros(33)}

    print("🚀 Starting analysis...")
    
    # 3. Process each file
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['segment_name']
        label = int(row['label']) # 0 or 1
        
        file_path = os.path.join(DATA_ROOT, filename)
        
        try:
            # Load data: (Frames, 33, 5)
            data = np.load(file_path)
            if data.size == 0: continue
            
            # Extract X and Y only: Shape (Frames, 33, 2)
            # data[:, :, 0] is X, data[:, :, 1] is Y
            xy_data = data[:, :, :2]
            
            # Filter NaNs (valid frames only)
            # We create a mask where True = Is a valid number
            valid_mask = ~np.isnan(xy_data).any(axis=2) # Shape (Frames, 33)
            
            # Check Out of Bounds (OOB)
            # Condition: x < 0 OR x > 1 OR y < 0 OR y > 1
            oob_condition = (xy_data[:, :, 0] < 0) | (xy_data[:, :, 0] > 1) | \
                            (xy_data[:, :, 1] < 0) | (xy_data[:, :, 1] > 1)
            
            # Apply valid mask to OOB check (ignore NaNs)
            final_oob = oob_condition & valid_mask
            
            # Sum up counts for this file
            # sum(axis=0) collapses frames -> gives count per landmark
            frames_count = valid_mask.sum(axis=0) 
            oob_count = final_oob.sum(axis=0)
            
            # Add to global totals
            total_frames[label] += frames_count
            oob_frames[label]   += oob_count
            
        except Exception as e:
            # print(f"Skipping {filename}: {e}")
            continue

    # 4. Calculate Percentages and Display
    print("\n" + "="*85)
    print(f"{'LANDMARK':<20} | {'NORMAL (0) OOB %':<18} | {'SEIZURE (1) OOB %':<18} | {'DIFF':<10}")
    print("="*85)

    results = []

    for i in range(33):
        name = LANDMARK_NAMES[i]
        
        # Calculate Normal %
        norm_total = total_frames[0][i]
        norm_pct = (oob_frames[0][i] / norm_total * 100) if norm_total > 0 else 0.0
        
        # Calculate Seizure %
        seiz_total = total_frames[1][i]
        seiz_pct = (oob_frames[1][i] / seiz_total * 100) if seiz_total > 0 else 0.0
        
        diff = seiz_pct - norm_pct
        
        print(f"{name:<20} | {norm_pct:>17.2f}% | {seiz_pct:>17.2f}% | {diff:>+9.2f}%")
        
        results.append({
            'Landmark': name,
            'Normal_OOB_Pct': norm_pct,
            'Seizure_OOB_Pct': seiz_pct,
            'Difference': diff
        })

    print("="*85)
    
    # Optional: Identify the most distinctive landmark
    best_indicator = sorted(results, key=lambda x: x['Difference'], reverse=True)[0]
    print(f"\n💡 Insight: The '{best_indicator['Landmark']}' leaves the frame {best_indicator['Difference']:.2f}% more often during seizures.")

if __name__ == "__main__":
    analyze_out_of_bounds()