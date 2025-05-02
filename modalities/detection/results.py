import json
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import ast

# ----------- Radar Parser for Stringified NumPy Arrays -----------
def parse_radar_string(radar_str):
    radar_str = radar_str.strip().replace("\n", " ")
    try:
        radar_array = ast.literal_eval(radar_str.replace("\n", " "))
        return radar_array.tolist()
    except Exception as e:
        print("[!] Failed to parse radar string:", e)
        return []

# ----------- General Position Extractor -----------
def extract_positions(data, modality_name):
    x_vals, y_vals = [], []

    # Radar special case (string that encodes array)
    if modality_name == "electronicDevices":
        print("hi")
        for entry in data:
            print(entry)
            if isinstance(entry, str):
                parsed = parse_radar_string(entry)
                for row in parsed:
                    print(row)
                    if len(row) >= 2:
                        x_vals.append(float(row[0]))
                        y_vals.append(float(row[1]))
    #else:
    ##    for detection_list in data:
    #        for detection in detection_list:
    #            if len(detection) >= 2:
    #                x_vals.append(float(detection[0]))
    #                y_vals.append(float(detection[1]))

    return x_vals, y_vals

# ----------- Plotting Function -----------
def plot_modalities_from_json(json_path):
    with open(json_path, "r") as f:
        scan_data = json.load(f)

    plt.figure(figsize=(10, 8))

    colors = [
        'red', 'blue', 'green', 'purple', 'orange',
        'cyan', 'magenta', 'yellow', 'brown', 'gray'
    ]
    markers = ['o', 's', '^', 'D', 'P', '*', 'x', 'v', '<', '>']

    for i, (modality, data) in enumerate(scan_data.items()):
        x, y = extract_positions(data, modality)
        if x and y:
            plt.scatter(x, y,
                        label=modality,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        alpha=0.75)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Detected Object Positions by Modality")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------- Example Usage -----------
if __name__ == "__main__":
    plot_modalities_from_json("scan_results/scan_result.json")