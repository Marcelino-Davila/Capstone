import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load CSV file
x = []
y = []
filename = r'D:\capstoneRoot\code\groundTruth.csv'

with open(filename, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if there is one
    for row in csv_reader:
        try:
            x.append(float(row[0]))  # Convert to float
            y.append(float(row[1]))  # Convert to float
        except ValueError:
            print(f"Skipping invalid row: {row}")

# Print to verify
for i in range(len(x)):
    print(x[i], y[i])

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=10, c='blue', label='Ground Truth')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title("Scatter Plot of Ground Truth Data")
plt.show()