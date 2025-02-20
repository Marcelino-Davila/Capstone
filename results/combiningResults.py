import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import List, Tuple

def find_coordinates_in_range(x_coords: List[float], y_coords: List[float],
                              x_range: Tuple[float, float],y_range: Tuple[float, float]) -> Tuple[List[float], List[float]]:

    # Convert inputs to numpy arrays for efficient filtering
    x_array = np.array(x_coords)
    y_array = np.array(y_coords)
    
    # Create boolean masks for x and y ranges
    x_mask = (x_array >= x_range[0]) & (x_array < x_range[1])
    y_mask = (y_array >= y_range[0]) & (y_array < y_range[1])
    
    # Combine masks to find points that satisfy both conditions
    combined_mask = x_mask & y_mask
    
    # Apply mask to get matching coordinates
    matching_x = x_array[combined_mask].tolist()
    matching_y = y_array[combined_mask].tolist()
    
    if len(matching_x) > 0:
        return True
    return False


class resultReader:
    def __init__(self):
        self.x = []
        self.y = []
    def readPoint(self,row):
        self.x.append(float(row[0]))
        self.y.append(float(row[1]))

class combiner:
    def __init__(self):
        self.x = 17
        self.y = 20
        self.grid = np.zeros((self.x,self.y))

    def compareModaliteis(self,radar,lidar,rgbd,rgbs,lwird,lwirs,x,y):
        target = 0
        if(radar):
            target+=1
        if(lidar):
            target+=1
        if(rgbd):
            target+=1
        if(rgbs):
            target+=1
        if(lwird):
            target+=1
        if(lwirs):
            target+=1
        if(target > 3):
            self.grid[x,y] = 1

class superReader:
    def __init__(self):
        self.groundTruth =  resultReader()
        self.radar = resultReader()
        self.lidar = resultReader()
        self.downRGB = resultReader()
        self.sideRGB = resultReader()
        self.downLWIR = resultReader()
        self.sideLWIR = resultReader()
    def readRow(self,type,row):
        if type == 0:
            self.groundTruth.readPoint(row)
        elif type == 1:
            self.radar.readPoint(row)
        elif type == 2:
            self.lidar.readPoint(row)
        elif type == 3:
            self.downRGB.readPoint(row)
        elif type == 4:
            self.sideRGB.readPoint(row)
        elif type == 1:
            self.downLWIR.readPoint(row)
        else:
            self.sideLWIR.readPoint(row)

fileNames = [r"groundTruth.csv",r"clustered_points.csv",
             r"lidar.csv",r"downRGB.csv",r"sideRGB.csv",
             r"downLWIR.csv",r"sideLWIR.csv"]

def filter_similar_coordinates(coordinates, threshold):
    filtered_coordinates = []
    for x1, y1 in coordinates:
        is_similar = False
        for x2, y2 in filtered_coordinates:
            distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if distance < threshold:
                is_similar = True
                break
        if not is_similar:
            filtered_coordinates.append((x1, y1))
    return filtered_coordinates

allResults = superReader()
fusionCore = combiner()

for i in range(len(fileNames)):
    with open(fileNames[i], 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            allResults.readRow(i,row)

radarResult = False
lidarResult = False
downRGBResult = False
sideRGBResult = False
downLWIRResult = False
sideLWIRResult = False

for x in range(17):
    for y in range(20):
        x_range = (float(x),float(x+1))
        y_range = (float(y),float(y+1))
        lidarResult = find_coordinates_in_range(allResults.lidar.x,allResults.lidar.y,x_range,y_range)
        radarResult = find_coordinates_in_range(allResults.radar.x,allResults.radar.y,x_range,y_range)
        downRGBResult = find_coordinates_in_range(allResults.downRGB.x,allResults.downRGB.y,x_range,y_range)
        sideRGBResult = find_coordinates_in_range(allResults.sideRGB.x,allResults.sideRGB.y,x_range,y_range)
        downLWIRResult = find_coordinates_in_range(allResults.downLWIR.x,allResults.downLWIR.y,x_range,y_range)
        sideLWIRResult = find_coordinates_in_range(allResults.sideLWIR.x,allResults.sideLWIR.y,x_range,y_range)
        fusionCore.compareModaliteis(radarResult,lidarResult,downRGBResult,sideRGBResult,downLWIRResult,sideLWIRResult,x,y)

#llResults.lidar = filter_similar_coordinates(list(zip(allResults.lidar.x,allResults.lidar.y)),.5)
#allResults.downRGB = filter_similar_coordinates(list(zip(allResults.downRGB.x,allResults.downRGB.y)),.005) 
#allResults.sideRGB = filter_similar_coordinates(list(zip(allResults.sideRGB.x,allResults.sideRGB.y)),.005) 
#allResults.downLWIR = filter_similar_coordinates(list(zip(allResults.downLWIR.x,allResults.downLWIR.y)),.5) 
#allResults.sideLWIR = filter_similar_coordinates(list(zip(allResults.sideLWIR.x,allResults.sideLWIR.y)),.5)   

y_coords, x_coords = np.where(fusionCore.grid > 0)

# Scatter plot of active points
plt.scatter(x_coords, y_coords, c='red', marker='o', label="data fusion results")
plt.scatter(allResults.groundTruth.x, allResults.groundTruth.y, s=10, c='green', label='ground truth')

# Grid formatting
plt.xticks(np.arange(0, fusionCore.x, 1))
plt.yticks(np.arange(0, fusionCore.y, 1))
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("combined modalities")
plt.legend()
plt.gca().invert_yaxis() 
plt.show()

plt.figure(figsize=(16, 12))
plt.scatter(allResults.groundTruth.x, allResults.groundTruth.y, s=10, c='green', label='GT')
#plt.scatter(allResults.radar.x, allResults.radar.y, s=3, c='red', label='RADAR')
#plt.scatter(allResults.lidar.x, allResults.lidar.y, s=3, c='blue', label='LIDAR')
#plt.scatter(allResults.downRGB.x, allResults.downRGB.y, s=3, c='orange', label='RGBd')
#plt.scatter(allResults.sideRGB.x, allResults.sideRGB.y, s=3, c='blue', label='RGBs')
plt.scatter(allResults.downLWIR.x, allResults.downLWIR.y, s=3, c='blue', label='LWIRd')
plt.scatter(allResults.sideLWIR.x, allResults.sideLWIR.y, s=3, c='purple', label='LWIRs')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title("results")
plt.show()
