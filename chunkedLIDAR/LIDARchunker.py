import numpy as np
import scipy.io as sp
import csv
import os

# Load LiDAR data
pointCloud = sp.loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat")
profile = sp.loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_profile_2024_07_31.mat")

x_lidar = pointCloud['x_lidar'].flatten()  # Flatten arrays to 1D
y_lidar = pointCloud['y_lidar'].flatten()
z_lidar = pointCloud['z_lidar'].flatten()

# Define number of chunks
num_x_chunks = 10
num_y_chunks = 10

# Compute bin edges (ensure correct range)
x_min, x_max = np.min(x_lidar), np.max(x_lidar)
y_min, y_max = np.min(y_lidar), np.max(y_lidar)
x_bins = np.linspace(x_min, x_max, num_x_chunks + 1)
y_bins = np.linspace(y_min, y_max, num_y_chunks + 1)

# Digitize assigns points to bins (clip to valid range)
x_indices = np.clip(np.digitize(x_lidar, x_bins) - 1, 0, num_x_chunks - 1)
y_indices = np.clip(np.digitize(y_lidar, y_bins) - 1, 0, num_y_chunks - 1)

# Output directory
output_dir = r"D:\capstoneRoot\data\chunkedLIDAR"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

file_handles = {}
csv_writers = {}

# Fill chunks with corresponding (x, y, z) points
for i in range(num_x_chunks):
    file_path = os.path.join(output_dir, f"LIDARchunk_{i}.csv")
    file_handles[(i)] = open(file_path, "w", newline="")
    csv_writers[(i)] = csv.writer(file_handles[(i)])
    csv_writers[(i)].writerow(["x", "y","z"])  # Write header

# Convert each chunk list to a NumPy array
for idx in range(len(x_lidar)):
    xi, yi = x_indices[idx], y_indices[idx]
    csv_writers[(xi)].writerow([x_lidar[idx], y_lidar[idx], z_lidar[idx]])

# Print chunk sizes
# Close all file handles
for fh in file_handles.values():
    fh.close()

print("LiDAR data successfully chunked and saved to CSV files.")