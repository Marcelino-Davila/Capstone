import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv

# Load CSV file
points = []
xl = []
yl = []
i=0
with open(r"D:\capstoneRoot\code\results\lidar.csv", 'r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:

            
        print(row)
        xl.append(row[1])
        yl.append(row[0])
        

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
plt.scatter(xl, yl, s=10, c='blue', label='Filtered Points')
#plt.scatter(x, y, s=10, c='red', label='Ground Truth')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title("Filtered Clustered Points vs Ground Truth")
plt.show()
