import RGB 
import pandas as pd
import csv


csvPaths = [r"D:\capstoneRoot\code\usefulIamges\downlookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\downlookingRGB.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingLWIR.csv",
            r"D:\capstoneRoot\code\usefulIamges\sidelookingRGB.csv"]
images = pd.read_csv(csvPaths[0])
results = []
for image in images.iterrows():
    if(RGB.detectTarget(image[0])):
        results.append((image[1],image[2]))


resultsPaths = [r"D:\capstoneRoot\code\results\downRGB.csv",
                r"D:\capstoneRoot\code\results\sideRGB.csv",
                r"D:\capstoneRoot\code\results\downLWIR.csv",
                r"D:\capstoneRoot\code\results\sideLWIR.csv"]


with open(resultsPaths[0], mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in results:
        writer.writerow([image[0], image[1]])