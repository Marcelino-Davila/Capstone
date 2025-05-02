import numpy as np

def detectTarget(image, threshold=10, pixel_count=50):
    image = image[220:270,290:350]
    dark_pixels = np.sum(image < threshold)
    
    # Check for bright pixels
    bright_pixels = np.sum(image > 255 - threshold)
    
    # Return True if either dark or bright anomalies are detected
    return dark_pixels > pixel_count or bright_pixels > pixel_count
