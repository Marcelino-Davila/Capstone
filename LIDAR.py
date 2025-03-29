import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


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

def dbscan_clustering(points, eps=0.5, min_points=10):
    """Perform DBSCAN clustering on the point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    
    print(f"Found {max_label + 1} clusters")

    # Assign colors to clusters
    colors = plt.get_cmap("tab10")(labels / max_label if max_label > 0 else 1)
    colors[labels < 0] = [0, 0, 0, 1]  # Black for noise
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels

def visualize_clusters(pcd):
    """Visualize clustered point cloud."""
    o3d.visualization.draw_geometries([pcd])

# Load LiDAR data
h5_file = r"D:\capstoneRoot\code\chunkedLIDAR\lidar_chunks.h5"
chunk_y, chunk_x = 5, 6  # Select chunk
points = load_lidar_chunk(h5_file, chunk_y, chunk_x)
visualize_clusters(points)

if points is not None:
    clustered_pcd, labels = dbscan_clustering(points, eps=1.0, min_points=15)
    visualize_clusters(clustered_pcd)