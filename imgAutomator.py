import modalities.detection.RGB as RGB
import modalities.detection.LWIR as LWIR
import modalities.detection.wireDetection as WD
from pathlib import Path
import csv
import cv2

csvPaths = [r"D:\capstoneRoot\code\usefulIamges\downlookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\downlookingRGB.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingRGB.csv"]

def scan(modality,x1,x2,y1,y2):
    if modality == "downLWIR":
        path = csvPaths[0]
    if modality == "downRGB":
        path = csvPaths[1]
    if modality == "sideLWIR":
        path = csvPaths[2]
    if modality == "sideRGB":
        path = csvPaths[3]    
    if modality == "wireDown":
        path = csvPaths[1]
    if modality == "wireSide":
        path = csvPaths[3]
    with open(path, 'r', newline='') as file:
        images = csv.reader(file)
        results = []
        if(modality == "downLWIR" or modality == "sideLWIR"):
            for image in images:
                print(image[0])
                if((float(image[1]) > x1) and (float(image[1]) < x2) and (float(image[2]) > y1) and (float(image[2]) < y2)):
                    if(LWIR.detectTarget(cv2.imread(Path(str(image[0]))))):
                        results.append((image[1],image[2]))
        elif(modality == "wireDown" or modality == "wireSide"):
            for image in images:
                print("scanning ...")
                if((float(image[1]) > x1) and (float(image[1]) < x2) and (float(image[2]) > y1) and (float(image[2]) < y2)):
                    if(WD.detect_wires(str(image[0]))):
                        results.append((image[1],image[2]))
        else:
            for image in images:
                print("scanning ...")
                if((float(image[1]) > x1) and (float(image[1]) < x2) and (float(image[2]) > y1) and (float(image[2]) < y2)):
                    if(RGB.detectionAlgorithm(cv2.imread(str(image[0])))):
                        results.append((image[1],image[2]))
    
    return(results)