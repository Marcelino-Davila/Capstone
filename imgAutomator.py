import RGB 
import pandas as pd
import csv
import cv2

csvPaths = [r"D:\capstoneRoot\code\usefulIamges\downlookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\downlookingRGB.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingRGB.csv"]

with open(csvPaths[0], 'r', newline='') as file:
    images = csv.reader(file)
    results = []
    for image in images:
        if(RGB.detectTarget(cv2.imread(str(image[0])),50,50,640,512)):
            print("target")
            results.append((image[1],image[2]))


resultsPaths = [r"D:\capstoneRoot\code\results\downLWIR.csv",
                r"D:\capstoneRoot\code\results\downRGB.csv",
                r"D:\capstoneRoot\code\results\sideLWIR.csv",
                r"D:\capstoneRoot\code\results\sideRGB.csv"]


with open(resultsPaths[0], mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in results:
        writer.writerow([image[0], image[1]])