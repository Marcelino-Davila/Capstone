import laspy
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy
import numpy as np
from scipy.stats import binned_statistic_2d
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# === CONFIG ===
tile_size = 2.0          # in meters
overlap = 0.5            # optional buffer
grid_res = 0.2           # resolution for heightmap
dbscan_eps = 0.5         # DBSCAN neighborhood size
min_samples = 2          # DBSCAN cluster threshold

edge_lower_threshold = 10
edge_higher_threshold = 20

def stream_and_process_tiles(las_path):
    with laspy.open(las_path) as las_file:
        # Read the entire file once
        las = las_file.read()
        
        header = las_file.header
        min_x, max_x = header.mins[0], header.maxs[0]
        min_y, max_y = header.mins[1], header.maxs[1]

        x_starts = np.arange(min_x, max_x, tile_size - overlap)
        y_starts = np.arange(min_y, max_y, tile_size - overlap)

        clusters_all = []
        i=0
        print(f"Streaming {len(x_starts)*len(y_starts)} tiles...")
        for x0 in x_starts:
            for y0 in y_starts:
                x1, y1 = x0 + tile_size, y0 + tile_size
                i+=1
                print("tile",i)

                # Filter points within tile bounding box
                mask = (
                    (las.x >= x0) & (las.x <= x1) &
                    (las.y >= y0) & (las.y <= y1)
                )
                if np.count_nonzero(mask) == 0:
                    continue
                x = las.x[mask]
                y = las.y[mask]
                z = las.z[mask]
                points = np.vstack((x, y, z)).T

                edge_points, labels = process_tile(points, (x0, y0), grid_res, dbscan_eps, min_samples)
                if len(labels) > 0:
                    print("cluster found")
                    clusters_all.append((edge_points, labels))

        return clusters_all

# === 1. Load LAS file ===
def load_las(filepath):
    las = laspy.read(filepath)
    points = np.vstack((las.x, las.y, las.z)).T
    return points

# === 2. Tile the point cloud ===
def make_tile_points(points, tile_size, overlap):
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

# === 3. Process a tile: edge detection + DBSCAN ===
def process_tile(tile_points, tile_origin, grid_res, eps, min_samples):
    if len(tile_points) < 100:
        return np.empty((0, 3)), np.empty(0)  # skip sparse tiles

    x0, y0 = tile_origin
    local_x = tile_points[:, 0] - x0
    local_y = tile_points[:, 1] - y0
    z = tile_points[:, 2]

    # Create height map
    x_bins = int(tile_size / grid_res)
    y_bins = int(tile_size / grid_res)
    stat, _, _, _ = binned_statistic_2d(
        local_x, local_y, z, statistic='max', bins=[x_bins, y_bins]
    )
    height_map = np.nan_to_num(stat, nan=0.0)

    # Edge detection
    img = ((height_map - height_map.min()) / (height_map.ptp() + 1e-6) * 255).astype(np.uint8)
    edges = cv2.Canny(img, edge_lower_threshold, edge_higher_threshold)
    edge_pixels = np.argwhere(edges > 0)

    if len(edge_pixels) == 0:
        return np.empty((0, 3)), np.empty(0)

    # Convert edge pixels to 3D points
    edge_points = []
    for i, j in edge_pixels:
        x_center = x0 + i * grid_res
        y_center = y0 + j * grid_res
        mask = (
            (tile_points[:, 0] >= x_center - grid_res/2) & (tile_points[:, 0] <= x_center + grid_res/2) &
            (tile_points[:, 1] >= y_center - grid_res/2) & (tile_points[:, 1] <= y_center + grid_res/2)
        )
        edge_points.append(tile_points[mask])
    edge_points = np.vstack(edge_points)

    if len(edge_points) < min_samples:
        return np.empty((0, 3)), np.empty(0)

    # DBSCAN on edge points
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(edge_points)

    return edge_points, labels

# === 4. Run full pipeline ===
def detect_objects_from_las(filepath):
    points = load_las(filepath)
    tiles = make_tile_points(points, tile_size, overlap)
    all_clusters = []

    print(f"Processing {len(tiles)} tiles...")

    for tile_points, origin in tiles:
        edge_points, labels = process_tile(tile_points, origin, grid_res, dbscan_eps, min_samples)
        if len(labels) > 0:
            all_clusters.append((edge_points, labels))

    return all_clusters

# === 5. Visualization ===
def plot_clusters(clusters):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for points, labels in clusters:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='tab20', s=1)
    ax.set_title("Detected Object Clusters")
    plt.show()

# === RUN ===
if __name__ == "__main__":
    from scipy.stats import binned_statistic_2d
    las_file = "output_lidar.las"
    clusters = stream_and_process_tiles("output_lidar.las")
    plot_clusters(clusters)

# Use the same process_tile function from earlier
# You can now call:
# clusters = stream_and_process_tiles("output_lidar.las")
# plot_clusters(clusters)

