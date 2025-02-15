import filePathFinder as pf
import fileReader as fr
import sys
import csv


class usefulImages:
    def __init__(self,filePath, x, y):
        self.imagePath = filePath
        self.imageX = x
        self.imageY = y
    
#index, frame# X Y Z
if len(sys.argv) < 2:
    print("needs file for folder paths")
    exit(0)
filePathName = sys.argv[1]
pathList = pf.Paths(filePathName)
fileHandler = fr.FileHandler()

imagesDownRGB = []
imagesSideRGB = []
imagesDownLWIR = []
imagesSideLWIR = []

imageDataDownRGB = fileHandler.get_handler(pathList.downRGB[0])
imageDataSideRGB = fileHandler.get_handler(pathList.sideRGB[0])
imageDataDownLWIR = fileHandler.get_handler(pathList.downLWIR[0])
imageDataSideLWIR = fileHandler.get_handler(pathList.sideLWIR[0])

del imageDataDownRGB.data[0]
del imageDataSideRGB.data[0]
del imageDataDownLWIR.data[0]
del imageDataSideLWIR.data[0]

xDownRGB = imageDataDownRGB.data[0][2]
yDownRGB = imageDataDownRGB.data[0][3]

xSideRGB = imageDataSideRGB.data[0][2]
ySideRGB = imageDataSideRGB.data[0][3]

xDownLWIR = imageDataDownLWIR.data[0][2]
yDownLWIR = imageDataDownLWIR.data[0][3]

xSideLWIR = imageDataSideLWIR.data[0][2]
ySideLWIR = imageDataSideLWIR.data[0][3]
#-------------------------------------------------------------------------------- DOWNLOOKING RGB ------------------------------------------------------------------
for image in imageDataDownRGB.data:
    if(image[0] > 16207):
        break
    if(abs(float(xDownRGB)-float(image[2])) > 2) or (abs(float(yDownRGB)-float(image[3])) > 2):
        imagesDownRGB.append(usefulImages(r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\RGB\image_{}.png".format(image[1]),image[2],image[3]))
        xDownRGB = image[2]
        yDownRGB = image[3]
imagesDownRGB.sort(key=lambda img: float(img.imageX))
filtered_images = []
prev_x, prev_y = None, None
for image in imagesDownRGB:
    x, y = float(image.imageX), float(image.imageY)
    if prev_x is None or abs(x - prev_x) > 0.5 or abs(y - prev_y) > 0.5:
        filtered_images.append(image)
        prev_x, prev_y = x, y

with open(r"D:\capstoneRoot\code\usefulIamges\downlookingRGB.csv", mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in filtered_images:
        writer.writerow([image.imagePath, image.imageX, image.imageY])
#-------------------------------------------------------------------------------- SIDELOOKING RGB ------------------------------------------------------------------
for image in imageDataSideRGB.data:
    if(image[0] > 16207):
        break
    if(abs(float(xSideRGB)-float(image[2])) > 2) or (abs(float(ySideRGB)-float(image[3])) > 2):
        imagesSideRGB.append(usefulImages(r"D:\capstoneRoot\data\ASPIRE_forDistro\2 Sidelooking\RGB\image_{}.png".format(image[1]),image[2],image[3]))
        xSideRGB = image[2]
        ySideRGB = image[3]
imagesSideRGB.sort(key=lambda img: float(img.imageX))
filtered_images = []
prev_x, prev_y = None, None
for image in imagesDownRGB:
    x, y = float(image.imageX), float(image.imageY)
    if prev_x is None or abs(x - prev_x) > 0.5 or abs(y - prev_y) > 0.5:
        filtered_images.append(image)
        prev_x, prev_y = x, y

with open(r"D:\capstoneRoot\code\usefulIamges\sidelookingRGB.csv", mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in filtered_images:
        writer.writerow([image.imagePath, image.imageX, image.imageY])
#-------------------------------------------------------------------------------- DOWNLOOKING LWIR ------------------------------------------------------------------
for image in imageDataDownLWIR.data:
    if(image[0] > 16207):
        break
    if(abs(float(xDownLWIR)-float(image[2])) > 2) or (abs(float(yDownLWIR)-float(image[3])) > 2):
        imagesDownLWIR.append(usefulImages(r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\LWIR\image_{}.png".format(image[1]),image[2],image[3]))
        xDownLWIR = image[2]
        yDownLWIR = image[3]
imagesDownLWIR.sort(key=lambda img: float(img.imageX))
filtered_images = []
prev_x, prev_y = None, None
for image in imagesDownRGB:
    x, y = float(image.imageX), float(image.imageY)
    if prev_x is None or abs(x - prev_x) > 0.5 or abs(y - prev_y) > 0.5:
        filtered_images.append(image)
        prev_x, prev_y = x, y

with open(r"D:\capstoneRoot\code\usefulIamges\downlookingLWIR.csv", mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in filtered_images:
        writer.writerow([image.imagePath, image.imageX, image.imageY])
#-------------------------------------------------------------------------------- SIDELOOKING LWIR ------------------------------------------------------------------
for image in imageDataSideLWIR.data:
    if(image[0] > 16207):
        break
    if(abs(float(xSideLWIR)-float(image[2])) > 2) or (abs(float(ySideRGB)-float(image[3])) > 2):
        imagesSideLWIR.append(usefulImages(r"D:\capstoneRoot\data\ASPIRE_forDistro\2 Sidelooking\LWIR\image_{}.png".format(image[1]),image[2],image[3]))
        xSideLWIR = image[2]
        ySideLWIR = image[3]
imagesSideLWIR.sort(key=lambda img: float(img.imageX))
filtered_images = []
prev_x, prev_y = None, None
for image in imagesSideLWIR:
    x, y = float(image.imageX), float(image.imageY)
    if prev_x is None or abs(x - prev_x) > 0.5 or abs(y - prev_y) > 0.5:
        filtered_images.append(image)
        prev_x, prev_y = x, y

with open(r"D:\capstoneRoot\code\usefulIamges\sidelookingLWIR.csv", mode="w", newline="") as file:
    file.truncate(0)
    writer = csv.writer(file)
    for image in filtered_images:
        writer.writerow([image.imagePath, image.imageX, image.imageY])