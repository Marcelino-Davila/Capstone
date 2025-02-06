import fileReader as ef
import matplotlib.pyplot as plt
import filePathFinder as pf
import sys
import json
import testRun as tst

def main():
    if len(sys.argv) < 2:
        print("needs file for folder paths")
        return
    with open('dataToOpen.json') as f:
        running = json.load(f)
    filePathName = sys.argv[1]
    pathList = pf.Paths(filePathName)
    test = tst.test(running, pathList)
    test.runTest()
    
if __name__ == "__main__":
    main()