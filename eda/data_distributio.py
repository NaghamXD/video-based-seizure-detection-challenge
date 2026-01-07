import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = 'train_data'  # Update if needed
SAMPLE_RATE = 1.0         # 1.0 = Use 100% of data. Set to 0.1 if you have huge data and run out of RAM.

def analyze_distribution():
    # 1. Load Data
    files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.npy')]
    print(f"📂 Found {len(files)} files. Loading data...")

    all_x = []
    all_y = []

    for filename in tqdm(files):
        try:
            # Load file
            data = np.load(os.path.join(DATA_ROOT, filename))
            # Shape (Frames, 33, 5)
            
            # Subsample if configured (to save memory on massive datasets)
            if SAMPLE_RATE < 1.0:
                indices = np.random.choice(data.shape[0], int(data.shape[0] * SAMPLE_RATE), replace=False)
                data = data[indices]

            # Flatten X and Y channels
            # X is channel 0, Y is channel 1
            xs = data[:, :, 0].flatten()
            ys = data[:, :, 1].flatten()

            # Filter NaNs immediately
            xs = xs[~np.isnan(xs)]
            ys = ys[~np.isnan(ys)]

            all_x.append(xs)
            all_y.append(ys)

        except Exception as e:
            continue

    # Concatenate into one giant array
    print("Converting to single array...")
    X_final = np.concatenate(all_x)
    Y_final = np.concatenate(all_y)

    print(f"📊 Analyzing {len(X_final):,} total points.")

    # 2. Calculate Statistics (IQR Method for Outliers)
    def get_stats(arr, name):
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # Count outliers based on IQR
        outliers_iqr = ((arr < lower_bound) | (arr > upper_bound)).sum()
        # Count outliers based on Frame boundaries (0-1)
        outliers_frame = ((arr < 0) | (arr > 1)).sum()
        
        print(f"\n--- {name} STATISTICS ---")
        print(f"Mean: {np.mean(arr):.4f} | Median: {np.median(arr):.4f}")
        print(f"Range: [{np.min(arr):.4f}, {np.max(arr):.4f}]")
        print(f"IQR boundaries (Statistical Normal): {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"Points outside 0-1 Frame: {outliers_frame:,} ({outliers_frame/len(arr)*100:.2f}%)")

    get_stats(X_final, "X AXIS")
    get_stats(Y_final, "Y AXIS")

    # 3. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Landmark Distribution Analysis ({len(files)} files)", fontsize=16)

    # Plot A: Histogram X
    axes[0,0].hist(X_final, bins=100, color='skyblue', log=True) # Log scale reveals tiny outlier bars
    axes[0,0].set_title("X Coordinate Distribution (Log Scale)")
    axes[0,0].set_xlabel("X Position (0-1 is normal)")
    axes[0,0].axvline(0, color='r', linestyle='--')
    axes[0,0].axvline(1, color='r', linestyle='--')
    axes[0,0].grid(True, alpha=0.3)

    # Plot B: Histogram Y
    axes[0,1].hist(Y_final, bins=100, color='salmon', log=True)
    axes[0,1].set_title("Y Coordinate Distribution (Log Scale)")
    axes[0,1].set_xlabel("Y Position (0-1 is normal)")
    axes[0,1].axvline(0, color='r', linestyle='--')
    axes[0,1].axvline(1, color='r', linestyle='--')
    axes[0,1].grid(True, alpha=0.3)

    # Plot C: Box Plots (The standard way to see outliers)
    axes[1,0].boxplot([X_final, Y_final], vert=False, tick_labels=['X', 'Y'], sym='k.') 
    axes[1,0].set_title("Box Plot (Dots = Outliers)")
    axes[1,0].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[1,0].axvline(1, color='r', linestyle='--', alpha=0.5)
    axes[1,0].grid(True)

    # Plot D: 2D Heatmap (Where are the points actually clustering?)
    # We limit this view to slightly outside 0-1 to see the main cluster clearly
    h = axes[1,1].hist2d(X_final, Y_final, bins=100, range=[[-0.5, 1.5], [-0.5, 1.5]], cmap='plasma', cmin=1)
    axes[1,1].set_title("2D Density Heatmap (Most Common Positions)")
    axes[1,1].set_xlabel("X")
    axes[1,1].set_ylabel("Y")
    axes[1,1].invert_yaxis() # Match image coordinates (0 at top)
    fig.colorbar(h[3], ax=axes[1,1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_distribution()