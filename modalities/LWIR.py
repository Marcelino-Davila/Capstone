from skimage.color import deltaE_ciede2000, rgb2lab
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

ImageSizeX = 1024
ImageSizeY = 1024

def processWindow(image,imWin,imageSizeX, imageSizeY, labImage):
    window = image[imWin.y1:imWin.y2,imWin.x1:imWin.x2]
    avgWindowColor = cv2.mean(window)[:3]
    background = removeWindow(image, imWin.x1,imWin.x2,imWin.y1)
    return colorCompare(avgWindowColor,background)    

def detectTarget(imageIn,x,y,imageSizeX,imageSizeY):
    image = removeShadow(imageIn)    
    imWin = Window(x,y,imageSizeX,imageSizeY)
    labImage = rgb2lab(image / 255.0)

    smallIm = cv2.resize(image,(1000,1000),interpolation=cv2.INTER_AREA)
    cv2.imshow("small",smallIm)
    #calculating average for regions
    i=0
    targets = []

    with ThreadPoolExecutor() as executor: 
        futures = []
        while (imWin.scanning):
            futures.append(executor.submit(processWindow,image,imWin,imageSizeX,imageSizeY,labImage))
            imWin.increment()
        results = [future.result() for future in futures]
        targets = [i for i(target,_) in enumerate(results) if target]

    return targets

def removeWindow(image,x1,x2,y1,y2,imgx,imgy):
    top = np.mean(image[0:y1,x1:x2])
    left = np.mean(image[0:imgy,0:x1])
    bottom = np.mean(image[y2:imgy,x1:x2])
    right = np.mean(image[0:imgy,x2:imgx])
    average = [(top[0] + left[0] + bottom[0] + right[0])/4,
               (top[1] + left[1] + bottom[1] + right[1])/4,
               (top[2] + left[2] + bottom[2] + right[2])/4]
    return average
    
def colorCompare(image,window):
    target = False
    difference = deltaE_ciede2000(image, window)
    if(difference> 65):
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
        elif (self.x2 >= self.imageSizeX):
            self.y1 += self.yInc
            self.y2 += self.yInc
            self.x1 = 0
            self.x2 = self.xInc
        else:
            self.x1 += self.xInc
            self.x2 += self.xInc

image_path = r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931079409.png"

# Load the image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Error: Image not found. Check the file path.")
startTime = time.time()
target = detectTarget(image,75,75,4096,3000)
endTime = time.time()
print(f"total time: {endTime-startTime}")
if(target):
    print("Target")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()