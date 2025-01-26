import fileReader as ef
import matplotlib.pyplot as plt
import filePathFinder as pf
import sys

def main():
    if len(sys.argv) < 2:
        return
    filePathName = sys.argv[1]
    pathList = pf.Paths(filePathName)
    fileHandler = ef.FileHandler()
    for file in pathList.groundTruth:
        groundTruth = fileHandler.get_handler(file)
    #for file in pathList.sideRGB:
    #    sideRGB = ef.file(file)
    #print(groundTruth.fileData)

    for data in groundTruth.data:
        data.printf()

if __name__ == "__main__":
    main()