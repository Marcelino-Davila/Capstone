import open3d as o3d
import fileReader as ef

fileReader = ef.FileHandler()


#super class to reference all the files and stuff when running tests 
class test:
    def __init__(self,parameters,filePaths):
        self.groundTruth = fileReader.get_handler(filePaths.groundTruth[0])
        self.sideRGBExcel = fileReader.get_handler(filePaths.sideRGB[0])
        filePaths.sideRGB.remove(filePaths.sideRGB[0])
        self.sideRGBImages = filePaths.sideRGB
        self.sideLWIR = fileReader.get_handler(filePaths.sideLWIR[0])
        filePaths.sideLWIR.remove(filePaths.sideLWIR[0])                                       
        self.downRGB = fileReader.get_handler(filePaths.downRGB[0])
        filePaths.downRGB.remove(filePaths.downRGB[0])
        self.downLWIR = fileReader.get_handler(filePaths.downLWIR[0])
        filePaths.downLWIR.remove(filePaths.downLWIR[0])
        self.LIDARPointCloud = fileReader.get_handler(filePaths.LIDAR[0])
        self.LIDARProfile = fileReader.get_handler(filePaths.LIDAR[1])
        self.LIDARPng= fileReader.get_handler(filePaths.LIDAR[2])
        self.xStart = parameters["xStart"]
        self.yStart = parameters["yStart"]
        self.xEnd = parameters["xEnd"]
        self.yEnd = parameters["yEnd"]
        self.precision = parameters["precision"]
        i=0
        #for index, row in self.file.iterrows():
        #    i+=1
        #    self.data.append(excelData(i,row['APPROVED FOR PUBLIC RELEASE'],row['Unnamed: 1'],row['Unnamed: 2'],row['Unnamed: 3']))
        

class excelData:
    def __init__(self,ID,name,x,y,z):
        self.number = ID
        self.ID = name
        self.x = x
        self.y = y
        self.z = z
    def printf(self):
        print(self.ID, self.ID, ": X",self.x, " Y", self.y, " Z", self.z)

class matPCData: #Variables x_lidar, y_lidar, z_lidar
    def __init__(self, ID,x,y,z):
        self.ID = ID
        self.x = x
        self.y = y
        self.z = z
    def printf(self):
        print(self.ID, self.ID, ": X",self.x, " Y", self.y, " Z", self.z)

class matProfileData: #Variables: x_grid, y_grid, z_avg_grid, z_avg_grid_out
    def __init__(self, ID,x,y,z):
        self.ID = ID
        self.data = o3d.geometry.PointCloud()
    def printf(self):
        print(self.ID, self.ID, ": X",self.x, " Y", self.y, " Z", self.z)
