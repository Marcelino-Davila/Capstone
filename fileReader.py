import pandas as pd
import scipy as sp
import PIL as img
import numpy as np
import open3d as o3d
import os

class FileHandler:
    def __init__(self):
        pass

    def get_handler(cls, filepath):
        ext = os.path.splitext(filepath)[1]
        ext = ext.lower()
        if ext == '.csv':
            return csv(filepath)
        elif ext == '.xlsx':
            return excel(filepath)
        elif ext == '.mat':
            return mat(filepath)
        elif ext == '.png':
            return png(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")


class excel:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = []
        self.file = pd.read_excel(filepath)

class csv:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = []
        self.file = pd.read_csv(filepath)
        i=0
        for index, row in self.file.iterrows():
            i+=1
            self.data.append((i,row['Unnamed: 0'],row['APPROVED FOR PUBLIC RELEASE'],row['Unnamed: 2'],row['Unnamed: 3']))

class mat:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = []
        self.file = sp.io.loadmat(filepath)

class png:
    def __init__(self,filepath):
        self.filepath = filepath