import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from scipy.ndimage import gaussian_filter, label
import csv
class LidarProcessingPipeline:
    def __init__(self, hdf5_file_path: str, 
                 height_threshold: float = 0.5,
                 smoothing_sigma: float = 1.0):
        self.hdf5_file_path = hdf5_file_path
        self.height_threshold = height_threshold
        self.smoothing_sigma = smoothing_sigma
        
        # Initialize basic metadata
        with h5py.File(self.hdf5_file_path, 'r') as f:
            chunks_group = f['lidar_data/chunks']
            self.num_chunks = chunks_group.attrs['num_chunks']
            self.chunk_boundaries = chunks_group.attrs['chunk_boundaries']
    
    def load_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(self.hdf5_file_path, 'r') as f:
            lidar_group = f['lidar_data']
            chunk_group = lidar_group['chunks'][f'chunk_{chunk_idx}']
            
            x_start = chunk_group.attrs['x_start']
            x_end = chunk_group.attrs['x_end']
            
            x_coords = lidar_group['x_coordinates'][x_start:x_end]
            y_coords = lidar_group['y_coordinates'][:]
            elevation = chunk_group['elevation'][:]
            
            return x_coords, y_coords, elevation
    
    def process_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, List[dict]]:
        # Load the chunk data
        x_coords, y_coords, elevation = self.load_chunk(chunk_idx)
        
        # Preprocess elevation data
        smoothed = gaussian_filter(elevation, sigma=self.smoothing_sigma)
        
        # Calculate gradients
        gradient_x = np.zeros_like(smoothed)
        gradient_y = np.zeros_like(smoothed)
        gradient_x[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
        gradient_y[1:-1, :] = smoothed[2:, :] - smoothed[:-2, :]
        
        # Compute gradient magnitude and detect edges
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_map = (gradient_magnitude > self.height_threshold).astype(np.uint8)
        
        # Find objects
        labeled_map, num_features = label(edge_map)
        
        # Extract object properties with enhanced coordinate information
        objects = []
        for i in range(1, num_features + 1):
            object_mask = labeled_map == i
            if np.sum(object_mask) >= 2:  # Minimum size threshold
                # Get indices where this object exists
                y_indices, x_indices = np.where(object_mask)
                
                # Store all coordinates that make up this object
                object_coordinates = {
                    # Get actual x,y coordinates for all points in the object
                    'point_coordinates': [
                        (x_coords[x], y_coords[y], elevation[y, x])
                        for y, x in zip(y_indices, x_indices)
                    ],
                    
                    # Store array indices for reference
                    'array_indices': list(zip(y_indices, x_indices)),
                    
                    # Calculate perimeter points (points that have at least one non-object neighbor)
                    'perimeter_coordinates': [
                        (x_coords[x], y_coords[y])
                        for y, x in zip(y_indices, x_indices)
                        if any(
                            0 <= y+dy < elevation.shape[0] and 
                            0 <= x+dx < elevation.shape[1] and 
                            labeled_map[y+dy, x+dx] != i
                            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0)]
                        )
                    ]
                }
                
                # Calculate additional object properties
                obj_props = {
                    'id': f'chunk_{chunk_idx}_obj_{i}',
                    'chunk_idx': chunk_idx,
                    
                    # Center point
                    'center': (
                        x_coords[int(np.mean(x_indices))],
                        y_coords[int(np.mean(y_indices))]
                    ),
                    
                    # Object size
                    'size': np.sum(object_mask),
                    
                    # Height information
                    'height_range': (
                        np.min(elevation[object_mask]),
                        np.max(elevation[object_mask])
                    ),
                    'mean_height': np.mean(elevation[object_mask]),
                    
                    # Spatial extent
                    'x_extent': (
                        x_coords[np.min(x_indices)],
                        x_coords[np.max(x_indices)]
                    ),
                    'y_extent': (
                        y_coords[np.min(y_indices)],
                        y_coords[np.max(y_indices)]
                    ),
                    
                    # Store all coordinate information
                    'coordinates': object_coordinates
                }
                objects.append(obj_props)
            else:
                labeled_map[object_mask] = 0
                
        return labeled_map, objects

    def print_object_details(self, objects: List[dict], max_points: int = 5) -> None:

        for obj in objects:
            print(f"\nObject {obj['id']}:")
            print(f"  Center: ({obj['center'][0]:.2f}, {obj['center'][1]:.2f})")
            print(f"  Size: {obj['size']} points")
            print(f"  Height range: {obj['height_range'][0]:.2f}m to {obj['height_range'][1]:.2f}m")
            print(f"  Mean height: {obj['mean_height']:.2f}m")
            print(f"  X extent: {obj['x_extent'][0]:.2f} to {obj['x_extent'][1]:.2f}")
            print(f"  Y extent: {obj['y_extent'][0]:.2f} to {obj['y_extent'][1]:.2f}")
            
            # Print first few point coordinates
            print(f"  Sample points (first {max_points}):")
            for x, y, z in obj['coordinates']['point_coordinates'][:max_points]:
                print(f"    ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Print number of perimeter points
            print(f"  Number of perimeter points: {len(obj['coordinates']['perimeter_coordinates'])}")


pipeline = LidarProcessingPipeline(
    hdf5_file_path=r'D:\capstoneRoot\code\chunkedLIDAR\lidar_chunks.h5',
    height_threshold=0.5,
    smoothing_sigma=1.0)
    
    # Process and visualize a specific chunk
chunk_idx = 1  # Process first chunk
#pipeline.visualize_chunk(chunk_idx)

with open(r"D:\capstoneRoot\code\results\lidar.csv", mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for i in range(10):
        map,objects = pipeline.process_chunk(i)
        previousObject = (-10,-10)
        for object in objects: 
            cord = object['center']
            if((previousObject[0]-cord[0]) < 1 and (previousObject[1]-cord[1]) < 1):
                next
            writer.writerow([cord[1],cord[0]])
            previousObject = cord
            
            