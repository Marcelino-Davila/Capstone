import laspy
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Parameters
window_size = 1.0  # 1m x 1m window
resolution = 0.01  # grid cell size (1cm)
output_dir = "lidar_images"
json_file = "grouped_fused_output.json"
las_file = "D:\capstoneRoot\code\output_lidar.las"  # Change to your .las file path

os.makedirs(output_dir, exist_ok=True)

# Load targets from JSON
with open(json_file, 'r') as f:
    targets = json.load(f)

# Flatten the nested list
target_coords = [item[0] for item in targets]

# Load LAS file
print("Loading LAS file...")
las = laspy.read(las_file)
points = np.vstack((las.x, las.y, las.z)).T

# Process each coordinate
for i, target in enumerate(target_coords):
    x_c, y_c = target["x"], target["y"]

    # Filter points within the window
    x_min, x_max = x_c - window_size / 2, x_c + window_size / 2
    y_min, y_max = y_c - window_size / 2, y_c + window_size / 2

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    sub_points = points[mask]

    if sub_points.shape[0] == 0:
        print(f"No points found around target {i}. Skipping...")
        continue

    # Generate height map
    x_img = ((sub_points[:, 0] - x_min) / resolution).astype(int)
    y_img = ((sub_points[:, 1] - y_min) / resolution).astype(int)
    grid_w = int(window_size / resolution)
    height_map = np.full((grid_w, grid_w), np.nan)

    for x_pix, y_pix, z in zip(x_img, y_img, sub_points[:, 2]):
        if 0 <= x_pix < grid_w and 0 <= y_pix < grid_w:
            current = height_map[y_pix, x_pix]
            height_map[y_pix, x_pix] = z if np.isnan(current) else max(z, current)

    # Plot and save image
    plt.figure(figsize=(4, 4))
    plt.imshow(height_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Height (m)')
    plt.title(f"Target {i} Height Map")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"target_{i}.png"))
    plt.close()

    print(f"Saved image for target {i}")

print("All images generated.")