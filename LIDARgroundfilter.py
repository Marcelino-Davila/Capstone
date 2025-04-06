import numpy as np
import h5py
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import sobel
from sklearn.cluster import DBSCAN

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

def filter_ground_by_relative_height(points, threshold=0.05):
    """
    Remove points that are within `threshold` meters of the estimated ground plane.
    This assumes ground is the lowest z-value in the point cloud.
    """
    ground_height = np.percentile(points[:, 2], 5)  # More robust than min()
    non_ground_mask = points[:, 2] > ground_height + threshold
    return points[non_ground_mask]