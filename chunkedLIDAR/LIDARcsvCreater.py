import csv
data = []
filenames = []
for i in range(10):
    with open(f"LIDARchunk_{i}.csv", mode='w', newline='') as file:
    # Create a csv.writer object
        writer = csv.writer(file)
    # Write data to the CSV file
        writer.writerows(data)
