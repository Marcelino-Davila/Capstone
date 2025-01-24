import fileReader as ef
import matplotlib.pyplot as plt
import filePathFinder as pf
import sys

def main():
    if len(sys.argv) < 2:
        return
    filePathName = sys.argv[1]
    pathList = pf.Paths(filePathName)
    print(pathList.groundTruth)
    for file in pathList.groundTruth:
        groundTruth = ef.file(file)
    for file in pathList.sideRGB:
        sideRGB = ef.file(file)


if __name__ == "__main__":
    main()