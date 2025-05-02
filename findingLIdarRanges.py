from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pointcloud_data = loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat")
profile_data = loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_profile_2024_07_31.mat")


x_lidar = pointcloud_data['x_lidar'].flatten()
y_lidar = pointcloud_data['y_lidar'].flatten()
z_lidar = pointcloud_data['z_lidar'].flatten()

# Sample data to reduce plot size (optional)
sample_size = 500000  # Adjust based on performance
indices = np.random.choice(len(x_lidar), sample_size, replace=False)

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_lidar[indices], y_lidar[indices], z_lidar[indices], c=z_lidar[indices], cmap='jet', marker='.', s=1)

#ax.set_xlabel('X')
##ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.set_title('LiDAR Point Cloud')
#plt.show()


x_grid = profile_data['x_grid'].flatten()
y_grid = profile_data['y_grid'].flatten()
z_avg_grid = profile_data['z_avg_grid']

# Combine into Nx3 array
points = np.vstack((x_lidar, y_lidar, z_lidar)).T

# Define voxel size
voxel_size = 0.5  # adjust depending on resolution and smoothing desired

# Compute voxel grid coordinates
voxel_indices = np.floor(points / voxel_size).astype(np.int32)

# Find unique voxels and average the points inside each
voxel_keys, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
voxel_points = np.zeros((len(voxel_keys), 3))

for i in range(3):
    voxel_points[:, i] = np.bincount(inverse_indices, weights=points[:, i]) / np.bincount(inverse_indices)

# Optional: subsample for visualization
sample_size = min(500000, len(voxel_points))
indices = np.random.choice(len(voxel_points), sample_size, replace=False)

# Plot voxelized point cloud
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(voxel_points[indices, 0], voxel_points[indices, 1], voxel_points[indices, 2],
           c=voxel_points[indices, 2], cmap='jet', marker='.', s=1)
ax.set_title("Voxel-Downsampled LiDAR Point Cloud")
plt.show()

# Create a meshgrid
X, Y = np.meshgrid(x_grid, y_grid)

# Plot 2D Surface
plt.figure(figsize=(12, 8))
plt.pcolormesh(X, Y, z_avg_grid.T, cmap='jet', shading='auto')  # Transpose if needed
plt.colorbar(label="Height (Z)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("LiDAR Profile Heatmap")
plt.show()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Downsample to improve performance
step = 10  # Adjust for speed vs. detail
X_sub = X[::step, ::step]
Y_sub = Y[::step, ::step]
Z_sub = z_avg_grid[::step, ::step]


#THIS IS THE PROBLEM
ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='jet', edgecolor='none')
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Height (Z)")
ax.set_title("3D LiDAR Profile Surface")
plt.show()