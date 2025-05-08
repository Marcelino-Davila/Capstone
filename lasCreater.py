from scipy.io import loadmat
import numpy as np
import laspy

def createLas():   
    mat_data = loadmat(r'D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat')
    x = mat_data['x_lidar'].flatten()
    y = mat_data['y_lidar'].flatten()
    z = mat_data['z_lidar'].flatten()

    # Stack to Nx3 array
    points = np.vstack((x, y, z)).T

    # Remove rows with NaN or inf
    valid_mask = np.all(np.isfinite(points), axis=1)
    points_clean = points[valid_mask]

    x, y, z = points_clean[:, 0], points_clean[:, 1], points_clean[:, 2]

    # Create a LAS header (you can customize point format and version)

    header = laspy.LasHeader(point_format=3, version="1.2")

    # Set offsets to the minimum of each axis
    header.offsets = [np.min(x), np.min(y), np.min(z)]

    # Set reasonable scales (e.g. 1 cm precision)
    header.scales = [0.01, 0.01, 0.01]  # scales must be > 0 and appropriate for your data

    # Create LAS data object
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    # Optional: add intensity, classification, etc. if available
    # las.intensity = intensity_array
    # las.classification = class_array

    # Write to file
    las.write("output_lidar.las")