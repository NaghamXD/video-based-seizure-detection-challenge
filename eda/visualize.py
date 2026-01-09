'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os

# --- CONFIGURATION ---
npy_file_name = 'child_100_1.npy'       # Name of the file you want to see
csv_file_path = 'train_data/train_data.csv'      # Path to your labels CSV
# If your .npy files are in a specific folder, add the path here:
npy_folder_path = 'train_data'                  # e.g., 'data/train/'

full_path = os.path.join(npy_folder_path, npy_file_name)

# --- LOAD LABEL (Optional) ---
try:
    df = pd.read_csv(csv_file_path)
    # Filter for the specific file
    row = df[df['segment_name'] == npy_file_name]
    if not row.empty:
        label_code = row['label'].values[0]
        label_text = "SEIZURE (1)" if label_code == 1 else "NORMAL (0)"
    else:
        label_text = "Label not found in CSV"
except Exception as e:
    label_text = "CSV not found or readable"
    print(f"Warning: Could not read labels: {e}")

# --- LOAD LANDMARKS ---
try:
    # Shape: (150, 33, 5) -> (frames, landmarks, [x, y, z, vis, pres])
    lmk_arr = np.load(full_path)
    print(f"Loaded {npy_file_name}")
    print(f"Shape: {lmk_arr.shape}")
except Exception as e:
    print(f"Error loading .npy file: {e}")
    exit()

# PRINT DATA VALUES
print("\n" + "="*40)
print(f"🔍 INSPECTION: First Frame (Frame 0)")
print("="*40)
# Check for empty files
if lmk_arr.shape[0] > 0:
    # Print the first 5 landmarks of the first frame
    # Columns are usually: [x, y, z, visibility, presence]
    print("Format: [x, y, z, visibility, presence]")
    print(lmk_arr[0, :, :]) 
    
    # Check bounds for the whole file
    min_x, max_x = np.nanmin(lmk_arr[:, :, 0]), np.nanmax(lmk_arr[:, :, 0])
    min_y, max_y = np.nanmin(lmk_arr[:, :, 1]), np.nanmax(lmk_arr[:, :, 1])
    min_z, max_z = np.nanmin(lmk_arr[:, :, 2]), np.nanmax(lmk_arr[:, :, 2])
    print("-" * 20)
    print(f"Global X Range: {min_x:.4f} to {max_x:.4f}")
    print(f"Global Y Range: {min_y:.4f} to {max_y:.4f}")
    print(f"Global Z Range: {min_z:.4f} to {max_z:.4f}")
    if min_x < 0 or max_x > 1 or min_y < 0 or max_y > 1:
        print("⚠️  NOTE: X and Y Values outside [0, 1] detected.")
        print("   (These points will be invisible in the plot due to fixed axis limits)")
else:
    print("File contains no frames.")
print("="*40 + "\n")

# Print first 5 frames and first 5 landmarks for inspection
# --- SETUP ANIMATION ---
# MediaPipe connections (skeleton lines)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Left Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Right Arm
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Right Leg
]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(f"{npy_file_name}\nLabel: {label_text}")
ax.set_xlim(0, 1)
ax.set_ylim(-1, 0) # Invert Y because screen coordinates start from top-left
ax.grid(True)

# Initialize plot elements
lines = [ax.plot([], [], 'c-', linewidth=2)[0] for _ in CONNECTIONS]
points = ax.scatter([], [], s=30, c='red')
text_frame = ax.text(0.05, -0.05, '', transform=ax.transAxes)

def update(frame_idx):
    # Get data for current frame
    # Shape of frame_data is (33, 5)
    frame_data = lmk_arr[frame_idx]
    
    # Check if frame has NaNs (missing data)
    if np.isnan(frame_data).any():
        text_frame.set_text(f"Frame: {frame_idx}/{len(lmk_arr)} (MISSING DATA)")
        # Hide everything
        points.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return lines + [points, text_frame]

    # Extract X and Y (and invert Y)
    xs = frame_data[:, 0]
    ys = -frame_data[:, 1]
    
    # Update Points
    # Stack x and y into a (33, 2) array
    points.set_offsets(np.c_[xs, ys])

    # Update Lines (Skeleton)
    for line, (start, end) in zip(lines, CONNECTIONS):
        line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
        
    text_frame.set_text(f"Frame: {frame_idx}/{len(lmk_arr)}")
    return lines + [points, text_frame]

# Create Animation
# interval=33 means approx 30 fps (1000ms / 30fps = 33.3ms)
ani = animation.FuncAnimation(fig, update, frames=len(lmk_arr), interval=33, blit=True)

print("Displaying animation... (Close window to exit)")
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os

# --- CONFIGURATION ---
npy_file_name = 'child_100_1.npy'       # Name of the file you want to see
csv_file_path = 'train_data/train_data.csv'      # Path to your labels CSV
npy_folder_path = 'train_data'                  # e.g., 'data/train/'

full_path = os.path.join(npy_folder_path, npy_file_name)

# --- LOAD LABEL (Optional) ---
try:
    df = pd.read_csv(csv_file_path)
    row = df[df['segment_name'] == npy_file_name]
    if not row.empty:
        label_code = row['label'].values[0]
        label_text = "SEIZURE (1)" if label_code == 1 else "NORMAL (0)"
    else:
        label_text = "Label not found in CSV"
except Exception as e:
    label_text = "CSV not found or readable"
    print(f"Warning: Could not read labels: {e}")

# --- LOAD LANDMARKS ---
try:
    # Shape: (150, 33, 5) -> (frames, landmarks, [x, y, z, vis, pres])
    lmk_arr = np.load(full_path)
    print(f"Loaded {npy_file_name}")
    print(f"Shape: {lmk_arr.shape}")
except Exception as e:
    print(f"Error loading .npy file: {e}")
    exit()

# --- 1. PRINT DATA VALUES ---
print("\n" + "="*40)
print(f"🔍 INSPECTION: First Frame (Frame 0)")
print("="*40)

if lmk_arr.shape[0] > 0:
    # Print the first 5 landmarks of the first frame
    print("Format: [x, y, z, visibility, presence] (First 5 joints)")
    print(lmk_arr[0, :, :]) 
    
    # Calculate Global Bounds (ignoring NaNs)
    # We use these to decide the camera zoom
    min_x = np.nanmin(lmk_arr[:, :, 0])
    max_x = np.nanmax(lmk_arr[:, :, 0])
    min_y = np.nanmin(lmk_arr[:, :, 1])
    max_y = np.nanmax(lmk_arr[:, :, 1])
    
    print("-" * 20)
    print(f"Global X Range: {min_x:.4f} to {max_x:.4f}")
    print(f"Global Y Range: {min_y:.4f} to {max_y:.4f}")
    
    if min_x < 0 or max_x > 1 or min_y < 0 or max_y > 1:
        print("⚠️  NOTE: Values outside [0, 1] detected.")
        print("   The plot axis will expand to show these points.")
else:
    print("File contains no frames.")
    exit()
print("="*40 + "\n")

# --- SETUP ANIMATION ---
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Left Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Right Arm
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Right Leg
]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(f"{npy_file_name}\nLabel: {label_text}")

# --- 2. AUTO-ZOOM LOGIC ---
# Instead of hardcoding 0 and 1, we use the min/max we calculated earlier.
# We add a 10% margin so points aren't stuck exactly on the edge.
margin_x = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 0.1
margin_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 0.1

ax.set_xlim(min_x - margin_x, max_x + margin_x)
# Invert Y logic: Max Y in data is "lowest" on screen, Min Y is "highest"
# Because screen coords (0,0) are top-left.
# So we map data_min -> top, data_max -> bottom
# Therefore limits should be (-max, -min) roughly.
ax.set_ylim(-(max_y + margin_y), -(min_y - margin_y)) 

ax.grid(True)
# ---------------------------

# Initialize plot elements
lines = [ax.plot([], [], 'c-', linewidth=2)[0] for _ in CONNECTIONS]
points = ax.scatter([], [], s=30, c='red')
text_frame = ax.text(0.05, -0.05, '', transform=ax.transAxes)

def update(frame_idx):
    frame_data = lmk_arr[frame_idx]
    
    if np.isnan(frame_data).any():
        text_frame.set_text(f"Frame: {frame_idx}/{len(lmk_arr)} (MISSING DATA)")
        points.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return lines + [points, text_frame]

    xs = frame_data[:, 0]
    ys = -frame_data[:, 1] # Invert Y for display
    
    points.set_offsets(np.c_[xs, ys])

    for line, (start, end) in zip(lines, CONNECTIONS):
        line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
        
    text_frame.set_text(f"Frame: {frame_idx}/{len(lmk_arr)}")
    return lines + [points, text_frame]

ani = animation.FuncAnimation(fig, update, frames=len(lmk_arr), interval=33, blit=True)

print("Displaying animation with AUTO-ZOOM... (Close window to exit)")
plt.show()
#'''