import open3d as o3d
import fileReader as ef
import scipy.io as sio

fileReader = ef.FileHandler()


#super class to reference all the files and stuff when running tests 
class test:
    def __init__(self,parameters,filePaths):
        self.parameters = parameters
        if(parameters["0groundTruth"]):
            self.groundTruth = fileReader.get_handler(filePaths.groundTruth[0])
        if(parameters["RGB"]):
            self.sideRGBExcel = fileReader.get_handler(filePaths.sideRGB[0])
            filePaths.sideRGB.remove(filePaths.sideRGB[0])
            self.sideRGBImages = filePaths.sideRGB
            self.downRGB = fileReader.get_handler(filePaths.downRGB[0])
            filePaths.downRGB.remove(filePaths.downRGB[0])
        if(parameters["LWIR"]):
            self.sideLWIR = fileReader.get_handler(filePaths.sideLWIR[0])
            filePaths.sideLWIR.remove(filePaths.sideLWIR[0])                                       
            self.downLWIR = fileReader.get_handler(filePaths.downLWIR[0])
            filePaths.downLWIR.remove(filePaths.downLWIR[0])
        if(parameters["LIDAR"]):
            self.LIDARPointCloud = fileReader.get_handler(filePaths.LIDAR[0])
            self.LIDARProfile = fileReader.get_handler(filePaths.LIDAR[1])
            self.LIDARPng= fileReader.get_handler(filePaths.LIDAR[2])
        if(parameters["RADAR"]):

            print()
            self.RADARSide = fileReader.get_handler(filePaths.RADARSide[0])
            self.RADARDown = fileReader.get_handler(filePaths.RADARDown[0])
            self.RADARIMG = fileReader.get_handler(filePaths.RADARDown[1])
        self.xStart = parameters["xStart"]
        self.yStart = parameters["yStart"]
        self.xEnd = parameters["xEnd"]
        self.yEnd = parameters["yEnd"]
        self.precision = parameters["precision"]
        i=0
        #for index, row in self.file.iterrows():
        #    i+=1
        #    self.data.append(excelData(i,row['APPROVED FOR PUBLIC RELEASE'],row['Unnamed: 1'],row['Unnamed: 2'],row['Unnamed: 3']))
    def runTest(self):
        for hh, hv, vh, vv, ximg, xstrip, y, z in zip(self.RADARSide.file["img_hh"],
                                                      self.RADARSide.file["img_hv"],
                                                      self.RADARSide.file["img_vh"],
                                                      self.RADARSide.file["img_vv"],
                                                      self.RADARSide.file["x_img"],
                                                      self.RADARSide.file["x_strip"],
                                                      self.RADARSide.file["y_img"],
                                                      self.RADARSide.file["z_img"]):
            print("hh", hh, "hv", hv, "vh", vh, "vv", vv, "ximg", ximg, "xstrip", xstrip, "y", y, "z", z)
        #for xlidar,ylidar,zlidar in zip(self.LIDARPointCloud.file["x_lidar"],self.LIDARPointCloud.file["y_lidar"],self.LIDARPointCloud.file["z_lidar"]):
        #    if(zlidar>=1):
        #        print("x",xlidar,"y",ylidar,"z",zlidar)

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
