import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
import datetime
from sklearn.metrics import confusion_matrix
from VSViG_Landmark import VSViG_Landmark_Base
from pathlib import Path

# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent
# ↑ adjust number of `.parent` depending on depth

# --- CONFIGURATION ---
# 1. Paths
DATA_ROOT = BASE_DIR / 'train_data'            
SPLIT_FOLDER = BASE_DIR / 'processed_landmarks/splits_stratified' 
CHECKPOINT_PATH = BASE_DIR / 'checkpoints_stratified/best_model.pth' # The model you want to test
OUTPUT_REPORT_DIR = BASE_DIR / 'test_reports_stratified'

# 2. Model Parameters (Must match what you trained with!)
MAX_FRAMES = 150
BATCH_SIZE = 32

# 3. Mapping (Must match training!)
MP_MAP = {
    0: [0], 1: [2], 2: [5], 3: [12], 4: [14], 5: [16], 
    6: [24], 7: [26], 8: [28], 9: [11], 10: [13], 11: [15], 
    12: [23], 13: [25], 14: [27]
}

# --- DATASET CLASS (Re-defined to ensure standalone execution) ---
class SeizureDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_frames=150):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        filename = row['segment_name']
        label = int(row['label'])
        file_path = os.path.join(self.root_dir, filename)

        try:
            raw_lmks = np.load(file_path)
            if raw_lmks.size == 0: raise ValueError
        except:
            raw_lmks = np.zeros((1, 33, 5), dtype=np.float32)

        mapped_data = np.zeros((self.max_frames, 15, 5), dtype=np.float32)
        mask = np.zeros((self.max_frames), dtype=np.float32)

        num_frames = min(self.max_frames, raw_lmks.shape[0])
        
        if num_frames > 0:
            current_raw = raw_lmks[:num_frames].copy()
            
            # Normalization
            current_raw[:, :, 0] -= 0.5
            current_raw[:, :, 1] -= 0.5
            current_raw[:, :, 0:3] /= 6.0

            # Mapping
            for target_idx, mp_indices in MP_MAP.items():
                mapped_data[:num_frames, target_idx, :] = current_raw[:, mp_indices[0], :]

            mask[:num_frames] = 1.0 
            mapped_data = np.nan_to_num(mapped_data, nan=0.0)

        return (torch.from_numpy(mapped_data), 
                torch.tensor(label, dtype=torch.float32), 
                torch.from_numpy(mask),
                filename) # RETURN FILENAME FOR REPORTING

def main():
    # 1. Setup
    if not os.path.exists(OUTPUT_REPORT_DIR): os.makedirs(OUTPUT_REPORT_DIR)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting Evaluation on {device}")
    print(f"🕒 Time: {timestamp}")

    # 2. Load Data
    test_csv = os.path.join(SPLIT_FOLDER, 'test_split.csv')
    if not os.path.exists(test_csv):
        print(f"❌ Error: Test split not found at {test_csv}")
        return

    test_dataset = SeizureDataset(test_csv, DATA_ROOT, MAX_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📂 Loaded {len(test_dataset)} test samples.")

    # 3. Load Model
    model = VSViG_Landmark_Base()
    model.to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"⚖️  Loading weights from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 4. Evaluation Loop
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_filenames = []
    
    print("running inference...")
    with torch.no_grad():
        for batch in test_loader:
            data, labels, mask, filenames = batch
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)

            outputs = model(data, mask=mask) # Logits
            probs = torch.sigmoid(outputs).cpu().numpy() # Probability (0-1)
            preds = (probs > 0.5).astype(int) # Binary Prediction (0 or 1)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_filenames.extend(filenames)

    # 5. Calculate Metrics (EXACTLY AS REQUESTED)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Extract TN, FP, FN, TP
    # Sklearn confusion matrix structure:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    # Avoid division by zero with small eps or check
    eps = 1e-7

    # SENSITIVITY (Recall) = TP / (TP + FN)
    sensitivity = tp / (tp + fn + eps)
    
    # SPECIFICITY = TN / (TN + FP)
    specificity = tn / (tn + fp + eps)
    
    # PRECISION = TP / (TP + FP)
    precision = tp / (tp + fp + eps)
    
    # F1-SCORE = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    
    # Accuracy (Bonus)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("\n📊 RESULTS:")
    print(f"   Sensitivity: {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   Precision:   {precision:.4f}")
    print(f"   F1-Score:    {f1:.4f}")
    print(f"   Accuracy:    {accuracy:.4f}")
    print("-" * 30)
    print(f"   Confusion Matrix:\n{cm}")
    print(f"   (TN={tn}, FP={fp}, FN={fn}, TP={tp})")

    # 6. Save Detailed Predictions (CSV)
    df_results = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'seizure_probability': all_probs
    })
    
    csv_filename = f"predictions_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_REPORT_DIR, csv_filename)
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Saved detailed predictions to: {csv_path}")

    # 7. Save Summary Report (JSON)
    report = {
        'timestamp': timestamp,
        'model_path': CHECKPOINT_PATH,
        'config': {
            'max_frames': MAX_FRAMES,
            'batch_size': BATCH_SIZE
        },
        'metrics': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
    }
    
    json_filename = f"report_{timestamp}.json"
    json_path = os.path.join(OUTPUT_REPORT_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"📄 Saved summary report to: {json_path}")

if __name__ == '__main__':
    main()