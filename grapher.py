import json
import matplotlib.pyplot as plt

groundTruthx = [3.65,8.8,6.25,11.25,6.25,3.75,8.65,6.2,11.25,13.12,13.12]
groundTruthy = [11.5,11.5,9.6,
                9.56,7.15,5.15,
                5.05,3.04,2.8
                ,9.89,11.8]

def plot_all_modalities(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(10, 8))
    colors = {
        "radar": "red",
        "visualObjectDown": "purple",
        "visualObjectSide": "cyan",
        "hotObjectsDown": "orange",
        "hotObjectsSide": "purple",
    }

    # ---- RADAR ----
    if "electronicDevices" in data:
        radar = data["electronicDevices"]
        x_vals = [entry["centroid_real"][0] for entry in radar]
        y_vals = [entry["centroid_real"][1] for entry in radar]
        plt.scatter(x_vals, y_vals, label="radar", color=colors["radar"], marker='o', alpha=0.8)

    # ---- OTHER MODALITIES ----
    for key in data:
        if key == "electronicDevices":
            continue

        x_vals = []
        y_vals = []
        for detection_list in data[key]:
            for detection in detection_list:
                if len(detection) >= 2:
                    x_vals.append(float(detection[0]))
                    y_vals.append(float(detection[1]))

        if x_vals and y_vals:
            plt.scatter(x_vals, y_vals, label=key, color=colors.get(key, "blue"), marker='x', alpha=0.6)
        for x, y in zip(groundTruthx, groundTruthy):
                plt.plot(x, y, marker='x', color='black', markersize=10, markeredgewidth=2)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Sensor Detections by Modality")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_all_modalities("code\scan_results\scan_result.json")