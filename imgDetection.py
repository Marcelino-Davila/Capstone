import cv2
import numpy as np

class BushDetector:
    def __init__(self):
        """
        Initialize the bush detection algorithm with default parameters.
        """
        # Color range for detecting green (typical for bushes)
        self.lower_green = np.array([30, 40, 40])   # Lower HSV green threshold
        self.upper_green = np.array([90, 255, 255]) # Upper HSV green threshold
        
        # Minimum and maximum area for bush contours
        self.min_bush_area = 500    # Minimum pixel area to be considered a bush
        self.max_bush_area = 50000  # Maximum pixel area to be considered a bush

    def detect_bushes(self, image_path):
        """
        Detect bushes in an image using color and contour analysis.
        
        :param image_path: Path to the input image
        :return: List of detected bush regions
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to HSV color space (better for color-based segmentation)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for green regions
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        bush_detections = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Check if the contour meets size criteria for a bush
            if self.min_bush_area < area < self.max_bush_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate compactness (how circular/blob-like the region is)
                perimeter = cv2.arcLength(contour, True)
                compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                bush_detections.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'compactness': compactness
                })
        
        return bush_detections

    def visualize_detections(self, image_path, output_path=None):
        """
        Visualize bush detections by drawing bounding boxes.
        
        :param image_path: Path to the input image
        :param output_path: Path to save the annotated image (optional)
        :return: Annotated image
        """
        # Read the image
        image = cv2.imread(image_path)
        
        # Detect bushes
        bush_detections = self.detect_bushes(image_path)
        
        # Draw bounding boxes
        for bush in bush_detections:
            x, y, w, h = bush['bbox']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with area and compactness
            label = f"Bush Area: {bush['area']:.0f}"
            cv2.putText(image, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save output image if path provided
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image

    def tune_detection(self, image_path):
        """
        Interactive method to help tune detection parameters.
        
        :param image_path: Path to the input image
        """
        # Read the image
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create trackbars for HSV thresholds
        cv2.namedWindow('Parameter Tuning')
        
        # Trackbars for lower HSV
        cv2.createTrackbar('H Low', 'Parameter Tuning', self.lower_green[0], 179, self.update_h_low)
        cv2.createTrackbar('S Low', 'Parameter Tuning', self.lower_green[1], 255, self.update_s_low)
        cv2.createTrackbar('V Low', 'Parameter Tuning', self.lower_green[2], 255, self.update_v_low)
        
        # Trackbars for upper HSV
        cv2.createTrackbar('H High', 'Parameter Tuning', self.upper_green[0], 179, self.update_h_high)
        cv2.createTrackbar('S High', 'Parameter Tuning', self.upper_green[1], 255, self.update_s_high)
        cv2.createTrackbar('V High', 'Parameter Tuning', self.upper_green[2], 255, self.update_v_high)
        
        # Trackbars for area thresholds
        cv2.createTrackbar('Min Area', 'Parameter Tuning', self.min_bush_area, 10000, self.update_min_area)
        cv2.createTrackbar('Max Area', 'Parameter Tuning', self.max_bush_area, 100000, self.update_max_area)
        
        while True:
            # Create green mask
            green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
            
            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Combine original and mask
            result = cv2.bitwise_and(image, image, mask=green_mask)
            
            # Display images
            cv2.imshow('Original', image)
            cv2.imshow('Green Mask', green_mask)
            cv2.imshow('Result', result)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Close all windows
        cv2.destroyAllWindows()

    # Callback methods for trackbars
    def update_h_low(self, val): self.lower_green[0] = val
    def update_s_low(self, val): self.lower_green[1] = val
    def update_v_low(self, val): self.lower_green[2] = val
    def update_h_high(self, val): self.upper_green[0] = val
    def update_s_high(self, val): self.upper_green[1] = val
    def update_v_high(self, val): self.upper_green[2] = val
    def update_min_area(self, val): self.min_bush_area = val
    def update_max_area(self, val): self.max_bush_area = val

def main():
    # Create bush detector
    detector = BushDetector()
    
    # Path to your image
    image_path = r"D:\capstoneRoot\data\ASPIRE_forDistro\1 Downlooking\RGB\image_2931074239.png"
    
    # Option 1: Quick detection and visualization
    detections = detector.detect_bushes(image_path)
    print(f"Found {len(detections)} potential bushes")
    
    # Visualize detections
    detector.visualize_detections(image_path, 'output_bush_detection.jpg')
    
    # Option 2: Interactive parameter tuning (uncomment if needed)
    # detector.tune_detection(image_path)

if __name__ == "__main__":
    main()