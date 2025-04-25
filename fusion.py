import laspy
import numpy as np
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


groundTruthx = [3.65,8.8,6.25,11.25,6.25,3.75,8.65,6.2,11.25,13.12,13.12]
groundTruthy = [11.5,11.5,9.6,
                9.56,7.15,5.15,
                5.05,3.04,2.8
                ,9.89,11.8]


def plot_grouped_fused_results(grouped_results):
    plt.figure(figsize=(10, 8))

    for i, group in enumerate(grouped_results):
        if not group:
            continue

        # Use first detection as group center (you can also average positions)
        group_center_x = group[0]["x"]
        group_center_y = group[0]["y"]
        group_radius = group[0].get("radius_2d", 1.0)
        group_match = any(entry["match"] for entry in group)

        # Circle color based on match status
        color = 'green' if group_match else 'red'
        alpha = 0.3 if group_match else 0.2

        # Draw filled circle
        circle = Circle(
            (group_center_x, group_center_y),
            radius=group_radius,
            color=color,
            alpha=alpha,
            edgecolor='black'
        )
        plt.gca().add_patch(circle)

        # Optional: draw group center
        plt.plot(group_center_x, group_center_y, 'ko', markersize=5)
        plt.plot(groundTruthx,groundTruthy,'r',markersize=3)

    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Grouped Fused Detections with Radar Radius")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def load_lidar_z_map(laz_path, resolution=0.25):
    las = laspy.read(laz_path)
    xs = las.x
    ys = las.y
    zs = las.z

    # Build a lookup dictionary or grid for fast z queries
    # Round (x, y) to the desired resolution
    coords = np.round(np.column_stack((xs, ys)) / resolution) * resolution
    z_map = {}
    for (x, y), z in zip(coords, zs):
        z_map[(x, y)] = z_map.get((x, y), z)  # Overwrite or average if needed

    return z_map

with open(r"code\scan_results\scan_result.json", "r") as f:
    detection_data = json.load(f)

radar_detections = detection_data["electronicDevices"]
rgb_detections = detection_data.get("visibleObjectDown", []) + detection_data.get("visibleObjectSide", [])

def is_rgb_nearby(radar_x, radar_y, rgb_data, threshold=0.25):
    for detection_list in rgb_data:
        for detection in detection_list:
            rgb_x, rgb_y = float(detection[0]), float(detection[1])
            if abs(radar_x - rgb_x) <= threshold and abs(radar_y - rgb_y) <= threshold:
                return True
    return False

def run_fusion(detection_data, lidar_map, resolution=0.25):
    radar_detections = detection_data["electronicDevices"]
    rgb_detections = detection_data.get("visibleObjectDown", []) + detection_data.get("visibleObjectSide", [])

    fused_results = []

    for radar in radar_detections:
        x, y = radar["centroid_real"][0], radar["centroid_real"][1]
        z_radar = radar["centroid_real"][2]

        grid_key = (round(x / resolution) * resolution, round(y / resolution) * resolution)
        z_lidar = lidar_map.get(grid_key, None)

        rgb_detected = is_rgb_nearby(x, y, rgb_detections)

        match = False
        if z_lidar is not None:
            if z_radar < (z_lidar-0.2):
                match = True
                description = "subsurface target"
            elif rgb_detected:
                match = True
                description = "metal target"
            elif radar['max_cluster_mag'] > 2.35e-05:
                match = True
                description = "metal target"
            else:
                match = False
                description = "no target"

        fused_results.append({
            "x": x,
            "y": y,
            "z_radar": z_radar,
            "z_lidar": z_lidar,
            "rgb_detected": rgb_detected,
            "description": description,
            "match": match
        })

    return fused_results

def group_fused_results(fused_results, eps=1.0):
    if not fused_results:
        return []

    coords = np.array([
        [entry["x"], entry["y"]]
        for entry in fused_results
    ])

    db = DBSCAN(eps=eps, min_samples=1)
    labels = db.fit_predict(coords)

    grouped = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(fused_results[idx])

    return list(grouped.values())

lidar_map = load_lidar_z_map(r"code\output_lidar.las")
fused = run_fusion(detection_data, lidar_map)

grouped_fused = group_fused_results(fused)

for i, group in enumerate(grouped_fused):
    print(f"\n🧩 Group {i+1} ({len(group)} detections):")
    for entry in group:
        status = "✅ Match" if entry["match"] else "❌ No match"
        print(f" - ({entry['description']}: {entry['x']:.2f}, {entry['y']:.2f}) | z_radar={entry['z_radar']:.2f} | {status}")

plot_grouped_fused_results(grouped_fused)