import numpy as np
from scipy.io import loadmat
from sklearn.cluster import DBSCAN

def radarDetection(x_range,y_range):
    print("RADAR")
    # Load the .mat file
    data = loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\2024_07_31_aspire_3d_sar_img_hanning.mat")
    img_hh = data['img_hh']
    x_img = data['x_img'].squeeze()
    y_img = data['y_img'].squeeze()
    z_img = data['z_img'].squeeze()

    # Convert to magnitude (HH polarization)
    img_data = np.abs(img_hh)

    # Thresholds
    mag1 = 1.225718e-05  # subsurface
    mag2 = 1.0           # surface

    # Coordinate ranges
    z_range1 = (0, 59)   # Subsurface: Python uses 0-based indexing
    z_range2 = (60, 99)  # Surface
    x_range = (0, 1199)
    y_range = (0, 1199)

    # Subsurface
    mask1 = (img_data > mag1)
    x1, y1, z1 = np.where(mask1)
    valid1 = (
        (z1 >= z_range1[0]) & (z1 <= z_range1[1]) &
        (x1 >= x_range[0]) & (x1 <= x_range[1]) &
        (y1 >= y_range[0]) & (y1 <= y_range[1])
    )
    x1, y1, z1 = x1[valid1], y1[valid1], z1[valid1]
    mags1 = img_data[x1, y1, z1]

    # Surface
    mask2 = (img_data > mag2)
    x2, y2, z2 = np.where(mask2)
    valid2 = (
        (z2 >= z_range2[0]) & (z2 <= z_range2[1]) &
        (x2 >= x_range[0]) & (x2 <= x_range[1]) &
        (y2 >= y_range[0]) & (y2 <= y_range[1])
    )
    x2, y2, z2 = x2[valid2], y2[valid2], z2[valid2]
    mags2 = img_data[x2, y2, z2]

    # Combine
    x_all = np.concatenate([x1, x2])
    y_all = np.concatenate([y1, y2])
    z_all = np.concatenate([z1, z2])
    magnitudes_all = np.concatenate([mags1, mags2])

    print(f"Total Detected Points: {len(x_all)}")

    # DBSCAN
    points = np.stack([x_all, y_all, z_all], axis=1)
    epsilon = 5
    MinPts = 3
    db = DBSCAN(eps=epsilon, min_samples=MinPts)
    idx = db.fit_predict(points)
    num_clusters = len(set(idx)) - (1 if -1 in idx else 0)
    print(f"Number of clusters (excluding noise): {num_clusters}")

    # Compute cluster features
    results = []
    for i in range(num_clusters):
        cluster_indices = np.where(idx == i)[0]
        cluster_points = points[cluster_indices]
        cluster_mags = magnitudes_all[cluster_indices]

        centroid_idx = cluster_points.mean(axis=0).round().astype(int)
        centroid_real = [
            float(x_img[centroid_idx[0]]),
            float(y_img[centroid_idx[1]]),
            float(z_img[centroid_idx[2]])
        ]

        distances_2d = np.linalg.norm(cluster_points[:, :2] - centroid_idx[:2], axis=1)
        farthest_idx = cluster_points[np.argmax(distances_2d)].round().astype(int)
        farthest_real = [
            float(x_img[farthest_idx[0]]),
            float(y_img[farthest_idx[1]]),
            float(z_img[farthest_idx[2]])
        ]

        radius_2d = float(np.linalg.norm(np.array(farthest_real[:2]) - np.array(centroid_real[:2])))

        max_mag_idx = np.argmax(cluster_mags)
        max_point_idx = cluster_points[max_mag_idx].round().astype(int)
        max_point_real_z = float(z_img[max_point_idx[2]])
        max_point_mag = float(cluster_mags[max_mag_idx])

        results.append({
            "centroid_real": centroid_real,
            "max_cluster_mag": float(cluster_mags.max()),
            "farthest_real": farthest_real,
            "radius_2d": radius_2d,
            "max_point_mag": max_point_mag,
            "max_point_real_z": max_point_real_z
        })

    return results
