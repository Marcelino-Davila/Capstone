import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sp
from sklearn.cluster import DBSCAN
from scipy.ndimage import sobel

pointCloud = sp.loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat")
profile = sp.loadmat(r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_profile_2024_07_31.mat")

x_grid = profile['x_grid']
y_grid = profile['y_grid']
z_avg_grid = profile['z_avg_grid']

x_lidar = pointCloud['x_lidar']
y_lidar = pointCloud['y_lidar']
z_lidar = pointCloud['z_lidar']



# Compute edges
edges_x = sobel(z_avg_grid, axis=0)
edges_y = sobel(z_avg_grid, axis=1)
edges = np.hypot(edges_x, edges_y)  # Combine gradients

# Threshold for object detection
edge_threshold = 2.0
object_edges = edges > edge_threshold

# Plot
plt.imshow(object_edges, cmap='gray', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower')
plt.colorbar(label='Edge Intensity')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Object Detection via Edge Detection')
plt.show()



X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z_avg_grid, cmap='terrain')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Height (Z)')
plt.title('LiDAR 3D Surface Map')
plt.show()

class cube:
    def __init__(self,cornerX,cornerY,cornerZ,L,W,H):
        self.cX = cornerX
        self.cY = cornerY
        self.cZ = cornerZ
        self.L = L
        self.W = W
        self.H = H


class scanner(cube):
    def __init__(self,startX,startY,startZ,L,W,H,xInc,yInc,zInc,xEnd,yEnd,zEnd):
        super().__init__(startX,startY,startZ,L,W,H)
        self.scanning = True
        self.xInc = xInc
        self.yInc = yInc
        self.zInc = zInc
        self.xEnd = xEnd
        self.yEnd = yEnd
        self.zEnd = zEnd