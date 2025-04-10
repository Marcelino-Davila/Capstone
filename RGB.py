from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.exposure import adjust_gamma

import matplotlib.pyplot as plt

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



width = 4096
height = 3000
    #width and height are swapped

xSize = 3000
ySize = 4096
offset = 200
xMin = int(xSize/2 - offset - 1)
xMax = int(xMin + 2*offset)
yMin = int(ySize/2 - offset - 1)
yMax = int(yMin + 2*offset)



def removeShadow(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image provided")

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # Merge and convert back to BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def increase_saturation(image, factor=1.5):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the saturation channel
    h, s, v = cv2.split(hsv)
    
    # Increase saturation, ensuring it doesn't exceed 255
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    
    # Merge channels back and convert to BGR
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    return enhanced_image

rgbInfo = {}
rgbInfo["width"] = 4096
rgbInfo["height"] = 3000
rgbInfo["pixelsPerMeter"] = 381
rgbInfo["ActualWidth"] = 10.752

RGBWindow = [(4096/2-50,3000/2+50),(4096/2+50,3000/2-50)]
LWIRWindow = [(640/2-50,512/2+50),(640/2+50,512/2-50)]

RGBDefault = [] 
LWIRDefault = []


def detectTarget(imageIn,type):
    if type == "RGB":
        window = RGBWindow
        default = RGBDefault
    else:
        window = LWIRWindow
        default = LWIRDefault

    
def colorCompare(image,window):
    target = False
    lab1 = rgb2lab(np.array([[image]]) / 255.0)
    lab2 = rgb2lab(np.array([[window]]) / 255.0)
    difference = deltaE_ciede2000(lab1[0, 0], lab2[0, 0])
    if(difference> 50):
        target = True
    return (target, difference)

'''
Idea 1: Boundary of the image:
        Find the bounds of the unique RGB values/ center of the image that is unique compared to other 
        photos
        Add the sliding window edge to it
Idea 2: Remove images with just dirt 
        Find file with only dirt, take the RGB average and save as a variable
        Compare center of the image to dirt, and if it is dirt return 0 for there being no object
'''

'''with open(csvPaths[0], 'r', newline='') as file:
    images = csv.reader(file)
    results = []
    for image in images:
        if(RGB.detectTarget(cv2.imread(str(image[0])),50,50,640,512)):
            print("target")
            results.append((image[1],image[2]))
image = csv.reader(file)'''

def detectionAlgorithm(image):
    #need to limit windowSize
    #image = image[xMin:xMax,yMin:yMax]
    [detection, newImage, colorAverage] = imageIsDirt(image)

    if(detection):
        edge_detected = windowEdgeDetection(newImage)
        checkEdges(newImage, edge_detected, colorAverage)
    #print(edge_detected)
    

    
    return 10 #arbitrary return
    
    '''image = image[xMin:xMax,yMin:yMax]#bright
    detection,objectColor = imageIsDirt(image)
    if(detection): 
        edges = windowEdgeDetection(image)
    return checkEdges(image,edges,objectColor)'''
    
def colorAverage(image):
    for i in range(xMin, xMax):
        for j in range(yMin, yMax):
            x = image[i, j]
            r = r + x[0]
            g = g + x[1]
            b = b + x[2]
    rAverage = r/(2*offset*2*offset)
    gAverage = g/(2*offset*2*offset)
    bAverage = b/(2*offset*2*offset)
    return rAverage/(gAverage+bAverage)

def imageIsDirt(image):
    #image = removeShadow(image)
    image = increase_saturation(image)
    image = adjust_gamma(image, gamma=6, gain=6)
    
    width = 4096
    height = 3000
    #width and height are swapped

    xSize = 3000
    ySize = 4096
    offset = 200
    xMin = int(xSize/2 - offset - 1)
    xMax = int(xMin + 2*offset)
    yMin = int(ySize/2 - offset - 1)
    yMax = int(yMin + 2*offset)

    r = 0
    g = 0
    b = 0

    for i in range(xMin, xMax):
        for j in range(yMin, yMax):
            x = image[i, j]
            r = r + x[0]
            g = g + x[1]
            b = b + x[2]
    rAverage = r/(2*offset*2*offset)
    gAverage = g/(2*offset*2*offset)
    bAverage = b/(2*offset*2*offset)
    
    if (rAverage/(gAverage+bAverage)) < 0.40:
        return False, image, 0 #(0,0,0)
    else:
        return True, image, rAverage/(gAverage+bAverage) #(rAverage,gAverage,bAverage)
    
def windowEdgeDetection(image):
    edge_roberts = filters.roberts(image[:, :, 0])
    edge_roberts = edge_roberts + filters.roberts(image[:, :, 1])
    edge_roberts = edge_roberts + filters.roberts(image[:, :, 2])

    '''print("NEW LINE")
    print(np.max(edge_roberts))
    print(edge_roberts[2500, 2500])'''

    edge_detected = []
    for i in range(xMin, xMax):
        for j in range(yMin, yMax):
            if (edge_roberts[i, j] > 1): #was originally 0.4 but had to many points
                #print(i, j)
                edge_detected.append((i, j))
    #print(np.max(edge_roberts))
    return edge_detected

def checkEdges(image, edges, average):
    objectDetected = []

    #print(len(edges))
    #print(range(len(edges)-1))
    
    #filter edges (not functional)
    '''for i in range(len(edges)-10):
        #print(i)
        cord1 = edges[i]
        xCord1 = cord1[0]
        yCord1 = cord1[1]
        cord2 = edges[i+1]
        #print(i+1)
        xCord2 = cord2[0]
        yCord2 = cord2[1]
        if ( xCord1 > (xCord2-offset/20) and xCord1 < (xCord2+offset/20) and yCord1 > (yCord2-offset/20) and yCord1 < (yCord2 +offset/20)):
            #<x1<x2+offset/20): #and y2-offset/20<y1<y2+offset/20):
            x = (0,0)
            edges[i] = x
    print(len(edges))'''
    

    for i in range(len(edges)):
        cord = edges[i]
        xCord = cord[0]
        yCord = cord[1]
        
        for j in range(10):
            
            for k in range(10):
                newAverage = image[xCord - 10 + j, yCord -10 + k, 0]/ (image[xCord - 10 + j, yCord -10 + k, 1] + image[xCord - 10 + j, yCord -10 + k, 2])
                #print(newAverage)
                if (newAverage >  6*average):
                    objectDetected.append((i, j))
                    break
            if (newAverage > 6*average):
               break
        #print(newAverage)
    #print(objectDetected)
    #print("current average", average)
    print("previous # of edges detected", len(edges))
    print("current # of edges detected", len(objectDetected))
    return objectDetected


'''image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067657.png") #bright
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931068917.png") #slightly dark
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067917.png") #dark image
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067825.png") #bright with metal cans not centered
print(imageIsDirt(image))

image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067661.png") 
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067662.png") 
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067663.png") 
print(imageIsDirt(image))

image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067702.png") 
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067708.png") 
print(imageIsDirt(image))
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931081269.png") 
print(imageIsDirt(image))


image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931077185.png") #metal cans
print(imageIsDirt(image))

timerstart = time.time()
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931072618.png") #plastic bottle
print(imageIsDirt(image))

timerend = time.time()
print(timerend- timerstart)'''

image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931067657.png")
image = cv2.imread(r"D:\Capstone\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931072618.png") #plastic bottle
x = detectionAlgorithm(image)

