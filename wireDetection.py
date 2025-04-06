import cv2
import numpy as np
from pathlib import Path


def detect_wires(
    image_path,
    target_color=np.array([135, 94, 87]),
    threshold=0.07,
    color_space="rgb",
    morph_size=1,
    min_contour_area=.2,
    output_dir="output",
    debug=True
):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the image
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read the image at {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None, None, None
    
    # Convert to float and normalize
    img_rgb_float = img_rgb.astype(np.float32) / 255.0
    target_color_float = target_color.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to reduce noise
    img_blurred = img_rgb_float
    #img_blurred = cv2.GaussianBlur(img_rgb_float, (blur_size, blur_size), 0)
    
    # Convert to the specified color space
    if color_space.lower() == 'hsv':
        # Convert RGB -> HSV
        img_processed = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2HSV)
        target_processed = cv2.cvtColor(np.uint8([[[target_color[0], target_color[1], target_color[2]]]]), 
                                         cv2.COLOR_RGB2HSV)[0][0].astype(np.float32) / 255.0
    elif color_space.lower() == 'lab':
        # Convert RGB -> LAB
        img_processed = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2LAB)
        target_processed = cv2.cvtColor(np.uint8([[[target_color[0], target_color[1], target_color[2]]]]), 
                                         cv2.COLOR_RGB2LAB)[0][0].astype(np.float32) / 255.0
    else:
        # Stay in RGB
        img_processed = img_blurred
        target_processed = target_color_float
    
    if debug:
        # Save the processed image for debugging
        debug_processed = (img_processed * 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/debug_processed.jpg", cv2.cvtColor(debug_processed, cv2.COLOR_RGB2BGR))
    
    # Compute color distance (Euclidean)
    distances = np.linalg.norm(img_processed - target_processed, axis=2)
    
    # Create binary mask using the threshold
    mask = distances < threshold
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask_cleaned = mask.astype(np.uint8) * 255
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    if debug:
        # Save the initial and cleaned masks for debugging
        cv2.imwrite(f"{output_dir}/debug_initial_mask.jpg", (mask.astype(np.uint8) * 255))
        cv2.imwrite(f"{output_dir}/debug_cleaned_mask.jpg", mask_cleaned)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
     
    # Create a visualization of the results
    highlighted = img_rgb.copy()
    
    # Draw valid contours with a green outline
    cv2.drawContours(highlighted, valid_contours, -1, (0, 255, 0), 2)
    
    # Highlight the detected wire pixels in red
    mask_bool = mask_cleaned > 0
    highlighted[mask_bool] = [255, 0, 0]  # Red for wire pixels
    
    # Save the result
    highlighted_bgr = cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)
    result_path = f"{output_dir}/highlighted_wires.jpg"
    cv2.imwrite(result_path, highlighted_bgr)
    
    print(f"Found {len(valid_contours)} wire segments. Result saved to {result_path}")
    
    return highlighted, mask_cleaned, valid_contours



image_path = r'D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931078138.png'
detect_wires(image_path)