from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.exposure import adjust_gamma
from skimage import filters
import cv2
import numpy as np
import math

width = 4096
height = 3000
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

def RGBToNormalized(image): #colorVectorNormalized
    for i in range(xMin, xMax):
        for j in range(yMin, yMax):
            x = image[i, j]
            r = x[0]
            g = x[1]
            b = x[2]
            colorNormalized = math.sqrt(r**2 + g**2 + b**2)
            r = r/colorNormalized
            g = g/colorNormalized
            b = b/colorNormalized
    return image

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

    edge_detected = []
    for i in range(xMin, xMax):
        for j in range(yMin, yMax):
            if (edge_roberts[i, j] > 1): #was originally 0.4 but had to many points
                #print(i, j)
                edge_detected.append((i, j))
    #print(np.max(edge_roberts))
    return edge_detected

def checkEdges(image, edges, average):
    objectDetected = False

    for i in range(len(edges)):
        cord = edges[i]
        xCord = cord[0]
        yCord = cord[1]
        
        for j in range(10):
            
            for k in range(10):
                newAverage = image[xCord - 10 + j, yCord -10 + k, 0]/ (image[xCord - 10 + j, yCord -10 + k, 1] + image[xCord - 10 + j, yCord -10 + k, 2])
                #print(newAverage)
                if (newAverage >  6*average):
                    objectDetected = True
                    break
            if (newAverage > 6*average):
               break
    return objectDetected

def checkEdgesWithSaturation(image, edges, average):
    r = 0
    g = 0
    b = 0
    for i in range(len(edges)):
        cord = edges[i]
        xCord = cord[0]
        yCord = cord[1]
        #print(xCord)
        
        for j in range(10):
            
            for k in range(10):
                x = image[xCord -5 + j, yCord -5 + k]
                r = r + x[0]
                g = g + x[1]
                b = b + x[2]
        rAverage = r/(10*10)
        gAverage = g/(10*10)
        bAverage = b/(10*10)
        if (rAverage/(gAverage+bAverage) > average):
                return True
    return False

def checkEdgesWithRGBNormalized(image, edges, average): 
    r = 0
    g = 0
    b = 0
    for i in range(len(edges)):
        cord = edges[i]
        xCord = cord[0]
        yCord = cord[1]
        #print(xCord)
        imageNormalized = RGBToNormalized(image)
        for j in range(10):
            for k in range(10):
                x = imageNormalized[xCord -5 + j, yCord -5 + k]
                r = r + x[0]
                g = g + x[1]
                b = b + x[2]
        rAverage = r/(10*10)
        gAverage = g/(10*10)
        bAverage = b/(10*10)
        if (rAverage > 0.8 or gAverage > 0.8 or bAverage > 0.8):
                return True
    return False

def detectionAlgorithm(image):
    [detection, newImage, colorAverage] = imageIsDirt(image)
    if(detection):
        edge_detected = windowEdgeDetection(newImage)
        return checkEdgesWithSaturation(newImage, edge_detected, colorAverage) #image or newImage depending on which checkEdges
        #return check = checkEdgesWithRGBNormalized(image, edge_detected, colorAverage)
    return False

