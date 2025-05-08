import os

#store all the file paths in filePaths.txt
#first line path to ground truth
#second line path to downlooking lwir
#third line path to downlooking rgb
#fourth line path to sidelooking lwir
#fifth line path to sidelooking rgb
#fourth line path to LIDAR


path = dict()

class Paths:
    def __init__(self, filePathsTxt):
        with open(filePathsTxt, 'r') as file:
            path["groundTruth"] = file.readline()
            path["groundTruth"] = path["groundTruth"].rstrip(" \n")
            path["downLWIR"] = file.readline()
            path["downLWIR"] = path["downLWIR"].rstrip(" \n")
            path["downRGB"] = file.readline()
            path["downRGB"] = path["downRGB"].rstrip(" \n")
            path["sideLWIR"] = file.readline()
            path["sideLWIR"] = path["sideLWIR"].rstrip(" \n")
            path["sideRGB"] = file.readline()
            path["sideRGB"] = path["sideRGB"].rstrip(" \n")
            path["LIDAR"] = file.readline()
            path["LIDAR"] = path["LIDAR"].rstrip(" \n")
            path["RADARDown"] = file.readline()
            path["RADARDown"] = path["RADARDown"].rstrip(" \n")
            path["RADARSide"] = file.readline()
            path["RADARSide"] = path["RADARSide"].rstrip(" \n")
        self.groundTruth = get_full_paths(path["groundTruth"])
        self.downLWIR = get_full_paths(path["downLWIR"])
        self.downRGB = get_full_paths(path["downRGB"])
        self.sideLWIR = get_full_paths(path["sideLWIR"])
        self.sideRGB = get_full_paths(path["sideRGB"])
        self.LIDAR = get_full_paths(path["LIDAR"])
        self.RADARSide = get_full_paths(path["RADARSide"])
        self.RADARDown = get_full_paths(path["RADARDown"])



def get_full_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths