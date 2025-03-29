import numpy as np
import h5py
import os
import math
from scipy.io import loadmat


chunk_boundaries_x = [(-1.5,-0.5,), (-0.5,1.5,), (1.5,2.5,),(2.5, 3.5,) ,(3.5,4.5,),
                      (4.5,5.5,), (5.5,6.5,), (6.5,7.5,), (7.5,8.5,),
                      (8.5,9.5,), (9.5,10.5,), (10.5,11.5,), (11.5,12.5,),
                      (12.5,13.5,), (13.5,14.5,), (14.5,15.5,), (15.5,16.5,), 
                      (16.5,17.5,), (17.5,18.5,), (18.5,19.5,), (19.5,20.5,), 
                      (20.5,21.5,), (21.5,22.5,),(22.5,23.5,)]

chunk_boundaries_y = [(-0.5,0.5), (0.5,1.5), (1.5,2.5), (2.5,3.5),
                      (3.5,4.5), (4.5,5.5), (5.5,6.5), (6.5,7.5),
                      (7.5,8.5), (8.5,9.5), (9.5,10.5), (10.5,11.5),
                      (11.5,12.5),(12.5,13.5)]

def chunk_matlab_lidar_by_coordinates(matlab_file, output_file, x_ranges, y_ranges, 
                                      compression="gzip", compression_opts=4):
    """
    Load lidar data from a MATLAB file, chunk it based on coordinate ranges, and save to HDF5.
    
    Parameters:
    -----------
    matlab_file : str
        Path to input MATLAB file containing x_lidar, y_lidar, z_lidar variables
    output_file : str
        Path to output HDF5 file
    x_ranges : list of tuples
        List of (min, max) ranges for x coordinates to define chunks
    y_ranges : list of tuples
        List of (min, max) ranges for y coordinates to define chunks
    compression : str
        Compression filter to use (gzip, lzf, or None)
    compression_opts : int
        Compression level (1-9 for gzip, ignored for lzf)
    """
    # Load data from MATLAB file
    print(f"Loading MATLAB file: {matlab_file}")
    mat_data = loadmat(matlab_file)
    
    # Extract lidar variables
    x_lidar = mat_data['x_lidar'].flatten()  # Ensure 1D arrays
    y_lidar = mat_data['y_lidar'].flatten()
    z_lidar = mat_data['z_lidar'].flatten()
    
    print(f"Loaded {len(x_lidar)} lidar points")
    print(f"X range: {x_lidar.min()} to {x_lidar.max()}")
    print(f"Y range: {y_lidar.min()} to {y_lidar.max()}")
    print(f"Z range: {z_lidar.min()} to {z_lidar.max()}")
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create a group for metadata
        metadata = f.create_group('metadata')
        metadata.attrs['point_count'] = len(x_lidar)
        metadata.attrs['x_range'] = [float(x_lidar.min()), float(x_lidar.max())]
        metadata.attrs['y_range'] = [float(y_lidar.min()), float(y_lidar.max())]
        metadata.attrs['z_range'] = [float(z_lidar.min()), float(z_lidar.max())]
        
        # Store the chunking information
        x_chunks_ds = metadata.create_dataset('x_chunks', data=np.array(x_ranges))
        y_chunks_ds = metadata.create_dataset('y_chunks', data=np.array(y_ranges))
        
        # Create a group for the chunks
        chunks_group = f.create_group('chunks')
        
        # Process each chunk based on coordinate ranges
        for y_idx, (y_min, y_max) in enumerate(y_ranges):
            for x_idx, (x_min, x_max) in enumerate(x_ranges):
                # Find points within this chunk
                mask = (
                    (x_lidar >= x_min) & (x_lidar <= x_max) &
                    (y_lidar >= y_min) & (y_lidar <= y_max)
                )
                
                # Get points in this chunk
                chunk_indices = np.where(mask)[0]
                num_points = len(chunk_indices)
                
                if num_points == 0:
                    print(f"Skipping empty chunk at y={y_min}-{y_max}, x={x_min}-{x_max}")
                    continue
                
                # Extract chunk data
                chunk_x = x_lidar[chunk_indices]
                chunk_y = y_lidar[chunk_indices]
                chunk_z = z_lidar[chunk_indices]
                
                # Create chunk group
                chunk_name = f'chunk_{y_idx}_{x_idx}'
                chunk_group = chunks_group.create_group(chunk_name)
                
                # Create datasets for x, y, z coordinates with compression
                chunk_group.create_dataset('x', data=chunk_x, compression=compression, compression_opts=compression_opts)
                chunk_group.create_dataset('y', data=chunk_y, compression=compression, compression_opts=compression_opts)
                chunk_group.create_dataset('z', data=chunk_z, compression=compression, compression_opts=compression_opts)
                
                # Store chunk metadata
                chunk_group.attrs['y_range'] = [y_min, y_max]
                chunk_group.attrs['x_range'] = [x_min, x_max]
                chunk_group.attrs['point_count'] = num_points
                
                print(f"Processed chunk {chunk_name}: y={y_min}-{y_max}, x={x_min}-{x_max}, points={num_points}")
        
    print(f"Successfully saved chunked data to {output_file}")


def create_grid_from_lidar_chunks(h5_file, grid_resolution=1.0, method='mean'):
    """
    Create a gridded elevation model from the chunked lidar data.
    
    Parameters:
    -----------
    h5_file : str
        Path to the chunked HDF5 file
    grid_resolution : float
        Size of grid cells in coordinate units
    method : str
        Method for handling multiple points in a cell ('mean', 'min', 'max')
        
    Returns:
    --------
    tuple: (grid, x_grid, y_grid)
        grid: 2D array of elevation values
        x_grid: 1D array of x coordinates for grid columns
        y_grid: 1D array of y coordinates for grid rows
    """
    with h5py.File(h5_file, 'r') as f:
        # Get metadata
        x_range = f['metadata'].attrs['x_range']
        y_range = f['metadata'].attrs['y_range']
        
        # Create grid
        x_grid = np.arange(x_range[0], x_range[1] + grid_resolution, grid_resolution)
        y_grid = np.arange(y_range[0], y_range[1] + grid_resolution, grid_resolution)
        grid = np.full((len(y_grid)-1, len(x_grid)-1), np.nan)
        
        # Create count grid for averaging
        count_grid = np.zeros_like(grid)
        
        # Process each chunk
        for chunk_name in f['chunks']:
            chunk = f[f'chunks/{chunk_name}']
            chunk_x = chunk['x'][()]
            chunk_y = chunk['y'][()]
            chunk_z = chunk['z'][()]
            
            # Process each point in the chunk
            for i in range(len(chunk_x)):
                # Find the grid cell for this point
                x_idx = int((chunk_x[i] - x_range[0]) / grid_resolution)
                y_idx = int((chunk_y[i] - y_range[0]) / grid_resolution)
                
                # Skip if outside the grid
                if x_idx < 0 or x_idx >= len(x_grid)-1 or y_idx < 0 or y_idx >= len(y_grid)-1:
                    continue
                
                # Update grid based on method
                if method == 'min':
                    if np.isnan(grid[y_idx, x_idx]) or chunk_z[i] < grid[y_idx, x_idx]:
                        grid[y_idx, x_idx] = chunk_z[i]
                elif method == 'max':
                    if np.isnan(grid[y_idx, x_idx]) or chunk_z[i] > grid[y_idx, x_idx]:
                        grid[y_idx, x_idx] = chunk_z[i]
                else:  # Default to mean
                    if np.isnan(grid[y_idx, x_idx]):
                        grid[y_idx, x_idx] = chunk_z[i]
                    else:
                        grid[y_idx, x_idx] = (grid[y_idx, x_idx] * count_grid[y_idx, x_idx] + chunk_z[i]) / (count_grid[y_idx, x_idx] + 1)
                    count_grid[y_idx, x_idx] += 1
        
        # Create centered coordinates for the grid cells
        x_centers = x_grid[:-1] + grid_resolution/2
        y_centers = y_grid[:-1] + grid_resolution/2
        
        return grid, x_centers, y_centers


def read_points_by_coordinates(h5_file, x_min, x_max, y_min, y_max):
    """
    Read lidar points within a specific coordinate range from the chunked HDF5 file.
    
    Parameters:
    -----------
    h5_file : str
        Path to the HDF5 file
    x_min, x_max, y_min, y_max : float
        Coordinate bounds of the region to read
        
    Returns:
    --------
    tuple: (x, y, z)
        Arrays of points within the specified region
    """
    x_points = []
    y_points = []
    z_points = []
    
    with h5py.File(h5_file, 'r') as f:
        # Find chunks that intersect with our region
        for chunk_name in f['chunks']:
            chunk = f[f'chunks/{chunk_name}']
            chunk_x_range = chunk.attrs['x_range']
            chunk_y_range = chunk.attrs['y_range']
            
            # Skip chunks that don't intersect our region
            if (chunk_y_range[1] <= y_min or chunk_y_range[0] >= y_max or 
                chunk_x_range[1] <= x_min or chunk_x_range[0] >= x_max):
                continue
            
            # Get chunk data
            chunk_x = chunk['x'][()]
            chunk_y = chunk['y'][()]
            chunk_z = chunk['z'][()]
            
            # Find points within the requested region
            mask = (
                (chunk_x >= x_min) & (chunk_x <= x_max) &
                (chunk_y >= y_min) & (chunk_y <= y_max)
            )
            
            if np.any(mask):
                x_points.append(chunk_x[mask])
                y_points.append(chunk_y[mask])
                z_points.append(chunk_z[mask])
    
    # Combine points from all relevant chunks
    if x_points:
        return np.concatenate(x_points), np.concatenate(y_points), np.concatenate(z_points)
    else:
        return np.array([]), np.array([]), np.array([])

# Example usage
matlab_file = r"A:\bombproject\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat"
output_file = r"A:\bombproject\code\chunkedLIDAR.h5"
# Chunk and save the data
chunk_matlab_lidar_by_coordinates(
    matlab_file,
    output_file,
    chunk_boundaries_x,
    chunk_boundaries_y,
    compression="gzip",
    compression_opts=4
)

# Example: Read points from a specific region
print("\nReading points from a specific region:")
region_x_min, region_x_max = 2.5, 3.5
region_y_min, region_y_max = 4.5, 5.5

x, y, z = read_points_by_coordinates(
    output_file,
    region_x_min,
    region_x_max,
    region_y_min,
    region_y_max
)

print(f"Found {len(x)} points in region")
if len(x) > 0:
    print(f"Z range in region: {z.min():.2f} to {z.max():.2f}")

# Example: Create a gridded elevation model
print("\nCreating gridded elevation model:")
grid, grid_x, grid_y = create_grid_from_lidar_chunks(output_file, grid_resolution=0.5)

print(f"Grid shape: {grid.shape}")
print(f"Grid X range: {grid_x.min()} to {grid_x.max()}")
print(f"Grid Y range: {grid_y.min()} to {grid_y.max()}")
print(f"Valid elevation cells: {np.sum(~np.isnan(grid))}")

print("\nHDF5 file structure:")
with h5py.File(output_file, 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"{name}/ (Group)")
        else:
            print(f"{name} (Dataset: {obj.shape}, {obj.dtype})")
    f.visititems(print_structure)