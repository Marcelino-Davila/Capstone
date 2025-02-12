from skimage.color import deltaE_ciede2000, rgb2lab
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectTarget(imageIn,x,y,imageSizeX,imageSizeY):
    image = removeShadow(imageIn)
    #split image to thirds for image averages
    imageLeft = image[0:3000,0:500]
    imageMiddle = image[0:3000,500:3596]
    imageRight = image[0:3000,3596:4096]

    smallIm = cv2.resize(image,(1000,1000),interpolation=cv2.INTER_AREA)
    cv2.imshow("small",smallIm)
    imWin = Window(x,y,imageSizeX,imageSizeY)
    #calculating average for regions
    avgLeft = cv2.mean(imageLeft)[:3] 
    avgMid = cv2.mean(imageMiddle)[:3] 
    avgRight = cv2.mean(imageRight)[:3]
    i=0
    targets = []
    while (imWin.scanning):

        window = image[imWin.y1:imWin.y2,imWin.x1:imWin.x2] 
        avgWindowColor = cv2.mean(window)[:3]
        cv2.imshow("window",window)

        if(imWin.x1 < 500):
            target, confidence = colorCompare(avgWindowColor,avgLeft)
        elif(imWin.x1 > 3596):
            target, confidence = colorCompare(avgWindowColor,avgRight)
        else:
            target, confidence = colorCompare(avgWindowColor,avgMid)

        if(target):
            i+=1
            print(f"FUCK THERES A BOMB {i}, x: {imWin.x1}, y: {imWin.y1}")
            targets.append(i)
            cv2.waitKey(0)
        imWin.increment()

    if(len(targets)>3):
        return True
    return False


def colorCompare(image,window):
    target = False
    lab1 = rgb2lab(np.array([[image]]) / 255.0)
    lab2 = rgb2lab(np.array([[window]]) / 255.0)
    difference = deltaE_ciede2000(lab1[0, 0], lab2[0, 0])
    print(f"diff: {difference}")
    if(difference> 35):
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

image_path = r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931079409.png"

# Load the image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Error: Image not found. Check the file path.")
target = detectTarget(image,50,50,4096,3000)
if(target):
    print("BOMB BOMB BOMB BOMB")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()