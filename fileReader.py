import pandas as pd
import scipy as sp
import PIL as img
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
        i=0
        for index, row in self.file.iterrows():
            i+=1
            self.data.append(excelData(i,row['APPROVED FOR PUBLIC RELEASE'],row['Unnamed: 1'],row['Unnamed: 2'],row['Unnamed: 3']))
class csv:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = []
        self.file = pd.read_csv(filepath)
        i=0
        for index, row in self.file.iterrows():
            i+=1
            self.data.append(excelData(i,row['APPROVED FOR PUBLIC RELEASE'],row['Unnamed: 1'],row['Unnamed: 2'],row['Unnamed: 3']))

class mat:
    def __init__(self,filepath):
        self.filepath = filepath

class png:
    def __init__(self,filepath):
        self.filepath = filepath

class excelData:
    def __init__(self,number,name,x,y,z):
        self.number = number
        self.name = name
        self.x = x
        self.y = y
        self.z = z
    def printf(self):
        print(self.number, self.name, ": X",self.x, " Y", self.y, " Z", self.z)
