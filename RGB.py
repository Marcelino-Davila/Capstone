from skimage.color import deltaE_ciede2000, rgb2lab
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def detectTarget(imageIn,x,y,imageSizeX,imageSizeY):
    #image = removeShadow(imageIn)
    image = imageIn
    imWin = Window(x,y,imageSizeX,imageSizeY)
    i=0
    targets = []
    windowNumber = 0
    while (imWin.scanning):
        windowNumber+=1
        window = image[imWin.y1:imWin.y2,imWin.x1:imWin.x2] 
        avgWindowColor = cv2.mean(window)[:3]
        SX1 = imWin.x1 - 200
        SX2 = imWin.x2 + 200
        SY1 = imWin.y1 - 200
        SY2 = imWin.y2 + 200
        if(SX1 < 0):
            SX1 = 0
        if(SX2 > imageSizeX):
            SX2 = imageSizeX
        if(SY1 < 0):
            SY1 = 0
        if(SY2 > imageSizeY):
            SY2 = imageSizeY
        surroundingWindow = image[SY1:SY2,SX1:SX2]
        background = back(surroundingWindow,imWin.x1,imWin.x2,imWin.y1,imWin.y2,SX2-imWin.x2,SY2-imWin.y2)
        target, confidence = colorCompare(avgWindowColor, background)

        if(target):
            i+=1
            targets.append(i)
        imWin.increment()
    print("i")
    if(len(targets)>0):
        return True
    return False

def back(background,x1,x2,y1,y2,imgx,imgy):
    top = cv2.mean(background[0:y1,x1:x2])
    left = cv2.mean(background[0:imgy,0:x1])
    bottom = cv2.mean(background[y2:imgy,x1:x2])
    right = cv2.mean(background[0:imgy,x2:imgx])
    average = [(top[0] + left[0] + bottom[0] + right[0])/4,
               (top[1] + left[1] + bottom[1] + right[1])/4,
               (top[2] + left[2] + bottom[2] + right[2])/4]
    return average
    
def colorCompare(image,window):
    target = False
    lab1 = rgb2lab(np.array([[image]]) / 255.0)
    lab2 = rgb2lab(np.array([[window]]) / 255.0)
    difference = deltaE_ciede2000(lab1[0, 0], lab2[0, 0])
    if(difference> 70):
        target = True
    return (target, difference)


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


        
class Window:
    def __init__(self,x,y,imageSizeX,imageSizeY):
        self.x1 = 0
        self.y1 = 0
        self.x2 = x
        self.y2 = y
        self.xInc = x
        self.yInc = y
        self.imageSizeX = imageSizeX
        self.imageSizeY = imageSizeY
        self.scanning = True

    def increment(self):
        if (self.x2 >= self.imageSizeX and self.y2 >= self.imageSizeY):
            self.scanning = False
        if (self.x2 >= self.imageSizeX):
            self.y1 += self.yInc
            self.y2 += self.yInc
            self.x1 = 0
            self.x2 = self.xInc
        self.x1 += self.xInc
        self.x2 += self.xInc

