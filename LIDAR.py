import numpy as np
import h5py
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import sobel
from sklearn.cluster import DBSCAN


def LiDAR_detection(x1,x2,y1,y2):
    pass


def load_lidar_chunk(h5_file, chunk_y, chunk_x):
    """Load a chunk of LiDAR data from an HDF5 file."""
    chunk_name = f'chunk_{chunk_y}_{chunk_x}'
    
    with h5py.File(h5_file, 'r') as f:
        if f'chunks/{chunk_name}' not in f:
            print(f"Chunk {chunk_name} does not exist.")
            return None
        
        chunk = f[f'chunks/{chunk_name}']
        x, y, z = chunk['x'][()], chunk['y'][()], chunk['z'][()]
    
    return np.vstack((x, y, z)).T  # Return Nx3 array


def generate_heightmap(points, grid_size=0.005):
    """Convert LiDAR points into a 2D heightmap."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Normalize coordinates
    x_min, y_min = x.min(), y.min()
    x, y = (x - x_min) / grid_size, (y - y_min) / grid_size
    x, y = x.astype(int), y.astype(int)

    # Create empty heightmap
    heightmap = np.full((x.max() + 1, y.max() + 1), np.nan)
    
    # Populate heightmap with max height values
    for i in range(len(z)):
        heightmap[x[i], y[i]] = max(heightmap[x[i], y[i]], z[i]) if not np.isnan(heightmap[x[i], y[i]]) else z[i]

    # Check for NaNs before replacing
    if np.isnan(heightmap).all():
        print("Warning: All height values are NaN. Check your LiDAR data or grid size.")
    else:
        heightmap[np.isnan(heightmap)] = np.nanmedian(heightmap)

    return heightmap


def apply_edge_detection(heightmap):
    """Apply Sobel edge detection on the heightmap."""
    dx = sobel(heightmap, axis=0, mode='constant')
    dy = sobel(heightmap, axis=1, mode='constant')
    edges = np.hypot(dx, dy)  # Compute gradient magnitude

    return edges

def detect_objects_dbscan(points, eps=5, min_samples=3, ground_height_threshold=-0.03):

    # 1. Filter out ground points (simple method, can be replaced with RANSAC plane fitting)
    non_ground_mask = points[:, 2] > ground_height_threshold
    non_ground_points = points[non_ground_mask]
    
    if len(non_ground_points) < min_samples:
        return []
    
    # 2. Run DBSCAN on x,y,z coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(non_ground_points)
    labels = clustering.labels_
    
    # 3. Extract each cluster as a potential object
    detected_objects = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
            
        # Get points for this cluster
        cluster_points = non_ground_points[labels == label]
        
        # Add basic filtering based on cluster size, height, etc.
        if len(cluster_points) >= min_samples:
            detected_objects.append(cluster_points)
    
    return detected_objects

def filter_ground_by_height(points, height_threshold=0.3):
    """Filter out points below a certain height threshold."""
    non_ground_mask = points[:, 2] > height_threshold
    return points[non_ground_mask]

def visualize_heightmap_and_edges(heightmap, edges):
    """Visualize heightmap and detected edges."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(heightmap, cmap='terrain')
    ax[0].set_title("Heightmap")
    
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Edge Detection")
    
    plt.show()


# Load LiDAR data
h5_file = r"D:\capstoneRoot\code\chunkedLIDAR\lidar_chunks.h5"
chunk_y, chunk_x = 7, 10
points = load_lidar_chunk(h5_file, chunk_y, chunk_x)
objects = detect_objects_dbscan(points)

if points is not None:
    heightmap = generate_heightmap(points)
    edges = apply_edge_detection(heightmap)
    visualize_heightmap_and_edges(heightmap, edges)