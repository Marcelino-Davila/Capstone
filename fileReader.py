import pandas as pd
import scipy as sp
import PIL as img

class file:
    def __init__(self, fileType="xlsx", fileName=" "):
        self.fileName = fileName
        if(fileType == "xlsx"):
            self.fileData = pd.read_excel(fileName)
        elif(fileType == "csv"):
            self.fileData = pd.read_csv(fileName)
        elif(fileType == "mat"):
            self.fileData = sp.io.loadmat(fileName)
        elif(fileType == "img"):
            self.fileData = img.Image.open(fileName)