import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv

# Load CSV file
df = pd.read_csv(r"D:\capstoneRoot\code\results\clustered_points.csv")  # Ensure CSV has 'X_Coordinate' and 'Y_Coordinate' columns

# Convert to NumPy array
points = df[['X_Coordinate', 'Y_Coordinate']].values

# Use KDTree to find neighbors within a 1-unit radius
tree = KDTree(points)
to_remove = set()
average_points = []

for i, point in enumerate(points):
    if i in to_remove:
        continue
    neighbors = tree.query_ball_point(point, 1.5)  # Get indices of neighbors
    neighbors.remove(i)  # Remove self from list
    to_remove.update(neighbors)  # Mark neighbors for removal
    
    # Compute average of neighbor coordinates
    avg_x = np.mean(points[neighbors + [i], 0])  # Average X values
    avg_y = np.mean(points[neighbors + [i], 1])  # Average Y values
    average_points.append((avg_x, avg_y))

# Remove redundant points
filtered_df = df.drop(index=list(to_remove)).reset_index(drop=True)

# Append average points
average_df = pd.DataFrame(average_points, columns=['X_Coordinate', 'Y_Coordinate'])
filtered_df = pd.concat([filtered_df, average_df], ignore_index=True)

# Load Ground Truth Points
x, y = [], []
filename = r'D:\capstoneRoot\code\results\groundTruth.csv'

with open(filename, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if there is one
    for row in csv_reader:
        try:
            x.append(float(row[0]))  # Convert to float
            y.append(float(row[1]))  # Convert to float
        except ValueError:
            print(f"Skipping invalid row: {row}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(filtered_df['X_Coordinate'], filtered_df['Y_Coordinate'], s=10, c='blue', label='Filtered Points')
plt.scatter(x, y, s=10, c='red', label='Ground Truth')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title("Filtered Clustered Points vs Ground Truth")
plt.show()
