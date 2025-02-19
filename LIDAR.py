import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import scipy.io as sp
from scipy.interpolate import griddata
from scipy.interpolate import griddata


def detectTarget(points):
    # Convert to NumPy array and ensure it's (N, 3)
    points = np.asarray(points, dtype=np.float64)
    
    if points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) shape, got {points.shape}")

    if points.shape[0] == 0:
        print("Warning: Empty point cloud chunk. Skipping...")
        return

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    points = np.asarray(pcd.points)  # Update points after downsampling

    if points.shape[0] == 0:
        print("Warning: No points after downsampling. Skipping...")
        return

    # Create grid for height map
    resolution = 0.5
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    x_grid = np.arange(x_min, x_max, resolution)
    y_grid = np.arange(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Interpolate height values using KDTree
    tree = KDTree(points[:, :2])
    _, idx = tree.query(grid_points, k=1)
    height_map = points[idx, 2].reshape(xx.shape)

    # Normalize for edge detection
    height_map_norm = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(height_map_norm, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(height_map_norm, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(sobelx**2 + sobely**2)

    # Threshold edges
    _, edges_binary = cv2.threshold(edges.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(edges_binary, cmap="gray")
    plt.title("Edge Detection Result")
    plt.colorbar()
    plt.show()




lidarProfile = sp.loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_profile_2024_07_31.mat")
x_grid = lidarProfile['x_grid']
y_grid = lidarProfile['y_grid']
z_avg_grid = lidarProfile['z_avg_grid']
# Open the HDF5 file
with h5py.File(r"D:\capstoneRoot\code\chunkedLIDAR\lidarPCData.h5", "r") as hf:
    # Initialize empty lists to store data
    x_list, y_list, z_list = [], [], []

    # Loop through chunks in order
    for chunk_name in sorted(hf.keys(), key=lambda x: int(x.split("_")[-1])):  # Ensure chunks are in order
        x_list.append(hf[chunk_name]["x"][:])  # Read and store
        y_list.append(hf[chunk_name]["y"][:])
        z_list.append(hf[chunk_name]["z"][:])

    # Concatenate all chunks back into single arrays
    x_lidar = np.concatenate(x_list)
    y_lidar = np.concatenate(y_list)
    z_lidar = np.concatenate(z_list)

    
X, Y = np.meshgrid(y_grid, x_grid) 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z_avg_grid, cmap='terrain')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Height (Z)')
plt.title('LiDAR 3D Surface Map')
plt.show()

# Verify the size
print(f"Reconstructed x_lidar size: {x_lidar.shape}")
print(f"Reconstructed y_lidar size: {y_lidar.shape}")
print(f"Reconstructed z_lidar size: {z_lidar.shape}")

X, Y = np.meshgrid(x_grid.squeeze(), y_grid.squeeze())

# Flatten the mesh for interpolation
grid_points = np.column_stack((X.ravel(), Y.ravel()))
grid_heights = z_avg_grid.ravel()

# Interpolate to get the expected height for each lidar point
z_expected = griddata(grid_points, grid_heights, (x_lidar, y_lidar), method='linear')

# Handle NaNs from interpolation
valid_mask = ~np.isnan(z_expected)
diff = np.zeros_like(z_lidar)
diff[valid_mask] = z_lidar[valid_mask] - z_expected[valid_mask]

# Identify object points (above the surface by threshold)
threshold = diff.mean() + 2 * diff.std()
objects = diff > threshold
object_points = np.column_stack((x_lidar[objects], y_lidar[objects], z_lidar[objects]))

# Apply DBSCAN to find clusters (potential objects)
clustering = DBSCAN(eps=1.5, min_samples=10).fit(object_points[:, :2])
labels = clustering.labels_

# Visualize Detected Objects
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# All LiDAR points in gray for context
ax.scatter(x_lidar, y_lidar, z_lidar, c='gray', s=1, alpha=0.3, label='LiDAR Points')

# Clustered Objects in Color
for cluster_id in np.unique(labels):
    if cluster_id == -1:
        color = 'black'  # Noise points
    else:
        color = plt.cm.tab10(cluster_id % 10)

    cluster_points = object_points[labels == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], s=10, label=f'Object {cluster_id}')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Height (Z)')
plt.title('3D Visualization of Detected Objects')
plt.legend()
plt.show()