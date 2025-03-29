from skimage.color import deltaE_ciede2000, rgb2lab
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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
