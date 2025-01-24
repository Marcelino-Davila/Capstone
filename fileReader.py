import pandas as pd
import scipy as sp
import PIL as img
import os

class file:
    def __init__(self, fileName=" "):
        fileType = os.path.splitext(fileName)[1]
        if(fileType == ".xlsx"):
            self.fileData = pd.read_excel(fileName)
        elif(fileType == ".csv"):
            self.fileData = pd.read_csv(fileName)
        elif(fileType == ".mat"):
            self.fileData = sp.io.loadmat(fileName)
        elif(fileType == ".img"):
            self.fileData = img.Image.open(fileName)