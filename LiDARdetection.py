import laspy
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_2d
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

# === CONFIG ===
tile_size = 2.0          # in meters
overlap = 0.5            # optional buffer
grid_res = 0.2           # resolution for heightmap
dbscan_eps = 0.9         # DBSCAN neighborhood size
min_samples = 8          # DBSCAN cluster threshold

edge_lower_threshold = 5
edge_higher_threshold = 10

def stream_process_efficient(las_path):
    """
    Process a LAS file without loading the entire file into memory at once.
    Uses a chunk-based approach for very large files.
    """
    all_centroids = []
    
    # Open the file to get header info only
    with laspy.open(las_path) as las_file:
        header = las_file.header
        min_x, max_x = header.mins[0], header.maxs[0]
        min_y, max_y = header.mins[1], header.maxs[1]
        
        # Create tile grid
        x_starts = np.arange(min_x, max_x, tile_size - overlap)
        y_starts = np.arange(min_y, max_y, tile_size - overlap)
        
        total_tiles = len(x_starts) * len(y_starts)
        print(f"Will process {total_tiles} tiles...")
    
    # Process each tile separately by reopening the file for each
    for i, x0 in enumerate(x_starts):
        for j, y0 in enumerate(y_starts):
            x1, y1 = x0 + tile_size, y0 + tile_size
            print(f"Processing tile {i*len(y_starts)+j+1}/{total_tiles} at ({x0:.1f}, {y0:.1f})")
            
            # Only load points for this specific tile
            tile_points = load_tile_points(las_path, x0, y0, x1, y1)
            
            if len(tile_points) < 100:
                continue
                
            # Process the tile
            centroids = process_tile(tile_points, (x0, y0), tile_size, grid_res, 
                                   dbscan_eps, min_samples)
            print(centroids)
            print("hi")
            # Store cluster centroids if found
            if centroids:
                all_centroids.extend(centroids)
                print(f"Found {len(centroids)} objects in this tile. Total: {len(all_centroids)}")
            
            # Force garbage collection to free memory
            tile_points = None
            import gc
            gc.collect()
            
    return all_centroids

def load_tile_points(las_path, x0, y0, x1, y1, chunk_size=1_000_000):
    """
    Load only points within a specific tile bounds.
    Process the file in chunks to minimize memory usage.
    """
    points_in_tile = []
    
    # Process the file in chunks
    with laspy.open(las_path) as las_file:
        for chunk in las_file.chunk_iterator(chunk_size):
            # Filter points within the tile
            mask = (
                (chunk.x >= x0) & (chunk.x <= x1) &
                (chunk.y >= y0) & (chunk.y <= y1)
            )
            
            if np.count_nonzero(mask) == 0:
                continue
                
            # Extract and store the points
            x = chunk.x[mask]
            y = chunk.y[mask]
            z = chunk.z[mask]
            
            chunk_points = np.vstack((x, y, z)).T
            points_in_tile.append(chunk_points)
            
            # Clear variables to free memory
            x, y, z, mask = None, None, None, None
    
    # Combine all chunks for this tile
    if not points_in_tile:
        return np.empty((0, 3))
        
    try:
        combined_points = np.vstack(points_in_tile)
        return combined_points
    except ValueError:
        # Handle empty arrays
        return np.empty((0, 3))

def process_tile(tile_points, tile_origin, tile_size, grid_res, eps, min_samples):
    if len(tile_points) < min_samples:
        return []  # Return empty list for tiles with too few points

    x0, y0 = tile_origin
    
    # Create local coordinates
    local_x = tile_points[:, 0] - x0
    local_y = tile_points[:, 1] - y0
    z = tile_points[:, 2]

    # Create height map
    x_bins = int(tile_size / grid_res)
    y_bins = int(tile_size / grid_res)
    
    # Handle empty bins with nan_to_num
    stat, _, _, _ = binned_statistic_2d(
        local_x, local_y, z, statistic='max', bins=[x_bins, y_bins]
    )
    height_map = np.nan_to_num(stat, nan=0.0)

    # Check if there's any variation in height
    if height_map.ptp() <= 0:
        return []  # No height variation, nothing to detect
        
    # Convert to 8-bit for processing
    img = ((height_map - height_map.min()) / height_map.ptp() * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # === IMPROVED EDGE DETECTION APPROACH ===
    
    # 1. Use Sobel operators for gradient calculation
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize for visualization and thresholding
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Apply adaptive thresholding for more strict edge detection
    # This works better than fixed thresholds for varying terrain
    height_threshold = np.mean(gradient_magnitude) + 1.5 * np.std(gradient_magnitude)
    edges = gradient_magnitude > height_threshold
    
    # 3. Morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # 4. Skeletonize edges for better centroids
    from skimage.morphology import skeletonize
    edges = skeletonize(edges).astype(np.uint8) * 255
    
    # Get edge pixels
    edge_pixels = np.argwhere(edges > 0)

    if len(edge_pixels) == 0:
        return []

    # Convert edge pixels back to 3D points
    edge_points = []
    for i, j in edge_pixels:
        # Convert grid coordinates to world coordinates
        x_center = x0 + (i * grid_res) + (grid_res / 2)
        y_center = y0 + (j * grid_res) + (grid_res / 2)
        
        # Find points within this grid cell
        mask = (
            (tile_points[:, 0] >= x_center - grid_res/2) & 
            (tile_points[:, 0] < x_center + grid_res/2) &
            (tile_points[:, 1] >= y_center - grid_res/2) & 
            (tile_points[:, 1] < y_center + grid_res/2)
        )
        
        cell_points = tile_points[mask]
        if len(cell_points) > 0:
            # Use maximum height point in each cell for better representation
            max_height_idx = np.argmax(cell_points[:, 2])
            edge_points.append(cell_points[max_height_idx])
    
    # Check if we found any edge points
    if not edge_points:
        return []
        
    try:
        edge_points = np.array(edge_points)
    except ValueError:
        return []

    if len(edge_points) < min_samples:
        return []

    # Improved clustering with HDBSCAN - better for varied density point clouds
    # If HDBSCAN is not available, fallback to DBSCAN with adaptive eps
    try:
        clusterer = HDBSCAN(min_cluster_size=min_samples, 
                           min_samples=min(5, min_samples),
                           cluster_selection_epsilon=eps)
        labels = clusterer.fit_predict(edge_points[:, :2])
    except ImportError:
        # Fallback to DBSCAN with calculated eps based on point density
        point_density = len(edge_points) / (tile_size * tile_size)
        adaptive_eps = max(eps, 0.5 / np.sqrt(point_density))
        clusterer = DBSCAN(eps=adaptive_eps, min_samples=min_samples)
        labels = clusterer.fit_predict(edge_points[:, :2])
    
    # Filter out noise points (label -1)
    unique_labels = np.unique(labels)
    cluster_centers = []
    
    # Calculate centroid for each cluster (excluding noise)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
            
        # Points belonging to this cluster
        cluster_mask = labels == label
        cluster_points = edge_points[cluster_mask]
        
        # Optional: Filter out small clusters
        min_cluster_size = 10  # Adjust as needed
        if len(cluster_points) < min_cluster_size:
            continue
            
        # Calculate weighted centroid (weighted by height)
        weights = cluster_points[:, 2] - np.min(cluster_points[:, 2]) + 0.1
        center_x = np.average(cluster_points[:, 0], weights=weights)
        center_y = np.average(cluster_points[:, 1], weights=weights)
        
        # Add to list of centroids
        cluster_centers.append((center_x, center_y))
        print("number of things", len(cluster_centers))
    
    return cluster_centers

def make_tile_points(points, tile_size, overlap):
    """Divide a point cloud into overlapping tiles"""
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    x_starts = np.arange(min_x, max_x, tile_size - overlap)
    y_starts = np.arange(min_y, max_y, tile_size - overlap)

    tiles = []
    for x0 in x_starts:
        for y0 in y_starts:
            x1, y1 = x0 + tile_size, y0 + tile_size
            mask = (
                (points[:, 0] >= x0) & (points[:, 0] <= x1) &
                (points[:, 1] >= y0) & (points[:, 1] <= y1)
            )
            tile = points[mask]
            if tile.size > 0:
                tiles.append((tile, (x0, y0)))
    return tiles

# === RUN ===
if __name__ == "__main__":
    las_file = "output_lidar.las"
    
    # Choose one of the processing methods:
    
    # 1. For standard files (whole file loaded at once)
    clusters = stream_process_efficient(las_file)
    print(clusters)
        
        # Debug visualization for a small region
        # Region format is (min_x, min_y, max_x, max_y)
        # visualize_height_map(las_file, region=None, resolution=grid_res)