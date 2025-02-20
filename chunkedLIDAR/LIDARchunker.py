import scipy.io as sp
import h5py as h5
import numpy as np
from typing import Tuple, List

class LidarDataManager:
    def __init__(self, matlab_file_path: str, hdf5_file_path: str, num_chunks: int = 10):
        """
        Manages LIDAR data storage and retrieval with chunking support.
        
        Parameters:
        -----------
        matlab_file_path : str
            Path to the input MATLAB file containing LIDAR data
        hdf5_file_path : str
            Path where the HDF5 file will be created
        num_chunks : int
            Number of chunks to divide the data into (default: 10)
        """
        self.matlab_file_path = matlab_file_path
        self.hdf5_file_path = hdf5_file_path
        self.num_chunks = num_chunks
        
    def calculate_chunk_boundaries(self, array_length: int) -> List[Tuple[int, int]]:
        """
        Calculates the start and end indices for each chunk.
        
        Parameters:
        -----------
        array_length : int
            Length of the array to be divided
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples for each chunk
        """
        chunk_size = array_length // self.num_chunks
        boundaries = []
        
        for i in range(self.num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_chunks - 1 else array_length
            boundaries.append((start_idx, end_idx))
            
        return boundaries
    
    def store_data(self):
        """
        Reads LIDAR data from MATLAB file and stores it in chunked HDF5 format.
        Each chunk contains a portion of the full dataset for efficient loading.
        """
        # Load the MATLAB data
        mat_data = sp.loadmat(self.matlab_file_path)
        x_lidar = mat_data['x_lidar'].flatten()
        y_lidar = mat_data['y_lidar'].flatten()
        z_lidar = mat_data['z_lidar']
        
        # Calculate chunk boundaries for x dimension
        x_chunks = self.calculate_chunk_boundaries(len(x_lidar))
        
        with h5.File(self.hdf5_file_path, 'w') as f:
            # Create main group for LIDAR data
            lidar_group = f.create_group('lidar_data')
            
            # Store full coordinate arrays
            lidar_group.create_dataset('x_coordinates', data=x_lidar,
                                     compression='gzip', compression_opts=4)
            lidar_group.create_dataset('y_coordinates', data=y_lidar,
                                     compression='gzip', compression_opts=4)
            
            # Create a group for chunks
            chunks_group = lidar_group.create_group('chunks')
            
            # Store metadata about chunking
            chunks_group.attrs['num_chunks'] = self.num_chunks
            chunks_group.attrs['chunk_boundaries'] = np.array(x_chunks)
            
            # Store each chunk separately
            for chunk_idx, (start_idx, end_idx) in enumerate(x_chunks):
                chunk_group = chunks_group.create_group(f'chunk_{chunk_idx}')
                
                # Store chunk metadata
                chunk_group.attrs['x_start'] = start_idx
                chunk_group.attrs['x_end'] = end_idx
                chunk_group.attrs['x_size'] = end_idx - start_idx
                chunk_group.attrs['y_size'] = len(y_lidar)
                
                # Store the chunk of elevation data
                chunk_group.create_dataset('elevation',
                                         data=z_lidar[start_idx:end_idx, :],
                                         chunks=True,
                                         compression='gzip',
                                         compression_opts=4)
    
    def load_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads a specific chunk of LIDAR data.
        
        Parameters:
        -----------
        chunk_idx : int
            Index of the chunk to load (0 to num_chunks-1)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x_coordinates, y_coordinates, and elevation data for the requested chunk
        """
        with h5.File(self.hdf5_file_path, 'r') as f:
            lidar_group = f['lidar_data']
            chunk_group = lidar_group['chunks'][f'chunk_{chunk_idx}']
            
            # Get chunk boundaries
            x_start = chunk_group.attrs['x_start']
            x_end = chunk_group.attrs['x_end']
            
            # Load relevant portions of coordinate arrays
            x_coords = lidar_group['x_coordinates'][x_start:x_end]
            y_coords = lidar_group['y_coordinates'][:]
            elevation = chunk_group['elevation'][:]
            
            return x_coords, y_coords, elevation

    
# Example usage

matlab_file = r"D:\capstoneRoot\data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat"
hdf5_file = r"D:\capstoneRoot\code\chunkedLIDAR\lidar_chunks.h5"
    
# Create manager and store data
manager = LidarDataManager(matlab_file, hdf5_file)
manager.store_data()
    
    # Example: Load the first chunk
x_chunk, y_chunk, z_chunk = manager.load_chunk(0)




print("happy")