import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

mat_data = scipy.io.loadmat("D:\\capstoneRoot\\data\\ASPIRE_forDistro\\2 Sidelooking\\img_cuda_2024_07_31_side_looking_hanning.mat")

sar_images = ['img_hh', 'img_hv', 'img_vh', 'img_vv']

def threshold_detection(image, threshold):
    return image > threshold  

def otsu_threshold(image):
    if image.ndim == 3:
        image = image[:, :, 0] 
    thresh_value = threshold_otsu(image)
    return image > thresh_value  

binary_results = {}

for img_name in sar_images:
    img = mat_data[img_name]  
    img_2D = img[:, :, 0]  

    threshold = np.mean(img_2D) + np.std(img_2D)
    binary_results[img_name] = otsu_threshold(img_2D)

    plt.figure()
    plt.imshow(binary_results[img_name], cmap='gray')
    plt.title(f"Otsu's Thresholding on {img_name} (First Channel)")
    plt.colorbar()
    plt.show()