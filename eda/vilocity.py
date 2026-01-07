import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  # <--- Essential for fixing folder paths

# ==========================================
# CONFIGURATION
# ==========================================
# We explicitly tell Python the folder name here
DATA_FOLDER = 'train_data'  
CSV_PATH = os.path.join(DATA_FOLDER, 'train_data.csv')

# ==========================================
# LOAD DATA
# ==========================================
# 1. Load the CSV from the 'train_data' folder
df = pd.read_csv(CSV_PATH)

# Extract Child ID from the filename
df['child_id'] = df['segment_name'].apply(lambda x: x.split('_')[1])

# MediaPipe connections (for skeleton drawing)
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (25, 27), (24, 26), (26, 28)            # Legs
]

def plot_skeleton(frame_landmarks, title="Skeleton"):
    plt.figure(figsize=(5, 5))
    
    x = frame_landmarks[:, 0]
    y = frame_landmarks[:, 1]
    
    # Invert Y for plotting (MediaPipe coordinates are often top-left origin)
    plt.scatter(x, -y) 
    
    # Draw lines
    for connection in SKELETON_CONNECTIONS:
        start_idx, end_idx = connection
        # Check for NaNs before drawing lines
        if not (np.isnan(x[start_idx]) or np.isnan(x[end_idx])):
            plt.plot([x[start_idx], x[end_idx]], [-y[start_idx], -y[end_idx]], 'r-')
            
    plt.title(title)
    plt.axis('equal')
    plt.show()

# ==========================================
# VISUALIZATION 1: SKELETON
# ==========================================
# Get the filename of the first seizure sample
seizure_filename = df[df['label'] == 1].iloc[0]['segment_name']

# --- CRITICAL FIX: Combine folder + filename ---
seizure_path = os.path.join(DATA_FOLDER, seizure_filename) 

print(f"Loading file from: {seizure_path}")
lmk = np.load(seizure_path)

# Plot the 50th frame
plot_skeleton(lmk[50], title=f"Seizure Frame - {seizure_filename}")


# ==========================================
# VISUALIZATION 2: DYNAMICS (VELOCITY)
# ==========================================
def plot_movement_dynamics(filename, label):
    # --- CRITICAL FIX: Add path inside the function ---
    file_path = os.path.join(DATA_FOLDER, filename) 
    
    data = np.load(file_path) # (150, 33, 5)
    
    # Extract Left Wrist (Index 15) and Right Wrist (Index 16) Y-coordinates
    left_wrist_y = data[:, 15, 1]
    right_wrist_y = data[:, 16, 1]
    
    frames = range(150)
    
    plt.figure(figsize=(12, 4))
    plt.plot(frames, left_wrist_y, label='Left Wrist Y')
    plt.plot(frames, right_wrist_y, label='Right Wrist Y')
    plt.title(f"Wrist Movement Over Time (Label: {label})")
    plt.xlabel("Frame (1-150)")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()
    
    # Calculate Velocity (magnitude of change)
    diff = np.diff(data[:, :, :3], axis=0) # Change in x,y,z
    velocity = np.sqrt(np.sum(diff**2, axis=2)) 
    avg_velocity = np.nanmean(velocity, axis=1) # Average movement
    
    plt.figure(figsize=(12, 4))
    plt.plot(avg_velocity, color='orange')
    plt.title(f"Average Body Velocity (Global Motion) - Label: {label}")
    plt.show()

# Compare a Seizure vs Non-Seizure
seizure_sample = df[df['label'] == 1].iloc[0]['segment_name']
normal_sample = df[df['label'] == 0].iloc[0]['segment_name']

print("--- SEIZURE SAMPLE ---")
plot_movement_dynamics(seizure_sample, "Seizure")

print("--- NORMAL SAMPLE ---")
plot_movement_dynamics(normal_sample, "Normal")