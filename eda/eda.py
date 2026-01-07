import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# --- CONFIGURATION ---
folder_path = 'train_data'
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Channels: 0=x, 1=y, 2=z, 3=vis, 4=pres
channel_names = ['x', 'y', 'z', 'visibility', 'presence']

# Initialize global accumulators
# We use infinity for min/max to ensure the first value overwrites them
global_min = np.full(5, np.inf)
global_max = np.full(5, -np.inf)
global_sum = np.zeros(5)
global_count = np.zeros(5)

print(f"Scanning {len(files)} files to calculate averages and ranges...")

for filename in files:
    try:
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        # data shape: (Frames, 33, 5)
        
        # Flatten to (Total_Points, 5)
        flat_data = data.reshape(-1, 5)
        
        # Filter out NaNs (rows where any value is NaN)
        # Note: Usually we check per column, but here we mask if the specific value is valid
        # Iterating per channel is safer for NaN handling
        
        for i in range(5):
            # Extract the specific channel column
            col_data = flat_data[:, i]
            
            # Remove NaNs from this specific column
            valid_vals = col_data[~np.isnan(col_data)]
            
            if valid_vals.size > 0:
                # Update Global Min
                current_min = np.min(valid_vals)
                if current_min < global_min[i]:
                    global_min[i] = current_min
                
                # Update Global Max
                current_max = np.max(valid_vals)
                if current_max > global_max[i]:
                    global_max[i] = current_max
                
                # Update Sums for Average (Weighted Average Strategy)
                global_sum[i] += np.sum(valid_vals)
                global_count[i] += valid_vals.size
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Print Report
print("\n" + "="*65)
print(f"{'Channel':<12} | {'Min':<10} | {'Max':<10} | {'Avg':<10} | {'Is 0-1?'}")
print("="*65)

for i, name in enumerate(channel_names):
    if global_count[i] > 0:
        c_min = global_min[i]
        c_max = global_max[i]
        
        # Calculate true global average
        c_avg = global_sum[i] / global_count[i]
        
        # strict check allowing small float errors
        is_01 = (c_min >= -0.01 and c_max <= 1.01)

        print(f"{name:<12} | {c_min:<10.4f} | {c_max:<10.4f} | {c_avg:<10.4f} | {str(is_01)}")
    else:
        print(f"{name:<12} | No valid data found")
print("="*65)


# Load the CSV
df = pd.read_csv('train_data/train_data.csv')

# Extract Child ID from the filename (assuming format 'child_{id}_{segment}.npy')
df['child_id'] = df['segment_name'].apply(lambda x: x.split('_')[1])

# 1. Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Class Distribution: Seizure (1) vs Normal (0)')
#plt.savefig('class_distribution.png')
#plt.show()

# 2. Clips per Child
plt.figure(figsize=(12, 5))
sns.countplot(x='child_id', hue='label', data=df)
plt.title('Number of Clips per Child (Split by Class)')
plt.xticks(rotation=90)
#plt.savefig('num_of_clips_per_child_split_by_class.png')
#plt.show()

print(f"Total Unique Children: {df['child_id'].nunique()}")