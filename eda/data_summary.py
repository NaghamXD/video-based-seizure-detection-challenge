import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
folder_path = 'train_data'       # Folder containing .npy files
labels_csv_path = 'train_data/train_data.csv' # Path to your labels CSV
output_excel_name = 'data_summary.xlsx'

# --- 1. LOAD LABELS ---
print(f"Loading labels from '{labels_csv_path}'...")
try:
    # Read the CSV containing labels
    labels_df = pd.read_csv(labels_csv_path, sep=',')
    
    # Create a quick lookup dictionary: { 'filename.npy': label_value }
    # Using a dict is faster for lookups inside the loop, or we can merge later.
    # Here we map 'segment_name' to 'label'.
    label_map = dict(zip(labels_df['segment_name'], labels_df['label']))
    
except Exception as e:
    print(f"⚠️ Warning: Could not load labels CSV. Error: {e}")
    label_map = {}

# --- 2. SCAN FILES ---
results = []
print(f"Scanning files in '{folder_path}'...")

# Get all .npy files
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

for filename in files:
    try:
        path = os.path.join(folder_path, filename)
        
        # Get the label from our map (default to 'Unknown' if not in CSV)
        # We try to match the filename exactly to the CSV 'segment_name'
        current_label = label_map.get(filename, 'Unknown')

        # Load the data
        # Shape is usually (Frames, Landmarks, Channels) -> (150, 33, 5)
        data = np.load(path)
        
        # Create a mask of valid frames (not all NaN)
        valid_frames_mask = ~np.isnan(data).all(axis=(1, 2))
        
        # Check if there is ANY valid data
        if not valid_frames_mask.any():
            # CASE: The file is completely empty (all NaNs)
            results.append({
                'File Name': filename,
                'Label': current_label,  # <--- Added Label here
                'Status': 'Has no data',
                'Start Frame': 'N/A',
                'Total Valid Frames': 0
            })
        else:
            # CASE: The file has data
            first_valid_index = np.argmax(valid_frames_mask)
            total_valid = np.sum(valid_frames_mask)
            
            results.append({
                'File Name': filename,
                'Label': current_label,  # <--- Added Label here
                'Status': 'Has Data',
                'Start Frame': first_valid_index,
                'Total Valid Frames': total_valid
            })
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        results.append({
            'File Name': filename,
            'Label': label_map.get(filename, 'Unknown'),
            'Status': f'Error: {str(e)}',
            'Start Frame': 'Error',
            'Total Valid Frames': 0
        })

# --- 3. SAVE TO EXCEL ---
if results:
    df = pd.DataFrame(results)
    
    # Sort by filename to make it tidy
    df = df.sort_values(by='File Name')
    
    # Save to Excel
    df.to_excel(output_excel_name, index=False)
    print(f"\n✅ Success! Report saved to '{output_excel_name}'")
    print(df.head()) 
else:
    print("No .npy files found.")