import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm  # Progress bar

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = 'train_data/train_data.csv'  # Path to your CSV
DATA_FOLDER = 'train_data'            # Path to folder containing .npy files (use '.' if same folder)

# ==========================================
# 1. LOAD METADATA
# ==========================================
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded CSV with {len(df)} records.")
except FileNotFoundError:
    print(f"Error: Could not find {CSV_FILE}. Please check the path.")
    exit()

# ==========================================
# 2. ANALYZE MISSINGNESS (WHOLE DATA)
# ==========================================
def analyze_clip(row):
    """Reads a file and counts NaN frames."""
    file_path = os.path.join(DATA_FOLDER, row['segment_name'])
    
    try:
        # Load data: Shape (150, 33, 5)
        data = np.load(file_path)
        
        # Check for NaNs
        # We check the first coordinate (x) of the first landmark. 
        # Usually if one is NaN, the whole frame is NaN.
        # axis=(1,2) checks if ANY landmark in a frame has ANY feature as NaN
        frames_is_nan = np.isnan(data).any(axis=(1, 2)) 
        nan_count = frames_is_nan.sum()
        
        return nan_count
    except Exception as e:
        return -1 # Mark as error

print("Starting analysis of all files... this may take a moment.")

# Apply the analysis with a progress bar
tqdm.pandas()
df['nan_count'] = df.progress_apply(analyze_clip, axis=1)

# Filter out files that weren't found (-1)
missing_files = df[df['nan_count'] == -1]
if len(missing_files) > 0:
    print(f"Warning: Could not find {len(missing_files)} files defined in CSV.")
    df = df[df['nan_count'] != -1]

# ==========================================
# 3. STATISTICAL SUMMARY
# ==========================================
print("\n--- MISSING DATA SUMMARY ---")
print(f"Total Clips Analyzed: {len(df)}")
print(f"Clips with ZERO missing frames: {len(df[df['nan_count'] == 0])} ({len(df[df['nan_count'] == 0])/len(df):.1%})")
print(f"Clips with >50% missing (75+ frames): {len(df[df['nan_count'] > 75])}")

# Group by Label
stats = df.groupby('label')['nan_count'].agg(['mean', 'median', 'std', 'max'])
print("\nMissing Frames by Class (0=No Seizure, 1=Seizure):")
print(stats)

# ==========================================
# 4. VISUALIZATION (CORRECTED)
# ==========================================
sns.set_style("whitegrid")

# Ensure labels are integers for consistency
df['label'] = df['label'].astype(int)

# Plot 1: Distribution of Missing Frames
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='nan_count', hue='label', element="step", bins=50, kde=True, palette={0: 'blue', 1: 'red'})
plt.title('Distribution of Missing Frames (Seizure vs Non-Seizure)')
plt.xlabel('Count of NaN Frames (Max 150)')
plt.ylabel('Number of Clips')
plt.legend(title='Label', labels=['Seizure', 'Normal'])
plt.savefig('missing_data_histogram.png')
plt.show()

# Plot 2: Boxplot (Fixed the error here)
plt.figure(figsize=(8, 6))
# We map 'hue' to 'label' to satisfy the new Seaborn requirements
sns.boxplot(x='label', y='nan_count', hue='label', data=df, palette={0: 'blue', 1: 'red'}, legend=False)
plt.title('Boxplot of Missing Frames by Class')
plt.xlabel('Label (0=Normal, 1=Seizure)')
plt.ylabel('NaN Frame Count')
plt.savefig('missing_data_boxplot.png')
plt.show()

# Plot 3: Scatter by Child ID
# Extract child_id
df['child_id'] = df['segment_name'].apply(lambda x: x.split('_')[1])
df['child_id_num'] = pd.to_numeric(df['child_id'], errors='coerce')

plt.figure(figsize=(14, 6))
# Using 'style' helps distinguish points if colors overlap
sns.scatterplot(data=df, x='child_id_num', y='nan_count', hue='label', style='label', alpha=0.6, palette={0: 'blue', 1: 'red'})
plt.title('Missing Frames per Child ID')
plt.xlabel('Child ID')
plt.ylabel('NaN Frame Count')
plt.legend(title='Label')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('missing_data_by_child.png')
plt.show()

print("\nAnalysis Complete. Plots saved.")