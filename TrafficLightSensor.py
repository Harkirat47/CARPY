import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Used claude.ai to edit/create some of this
class TrafficLightDetector:
    def __init__(self):
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        self.yellow_lower = np.array([15, 50, 50])
        self.yellow_upper = np.array([35, 255, 255])
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
    
    def load_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def preprocess_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        return hsv_blurred
    
    def detect_color_regions(self, hsv_image):
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        return red_mask, yellow_mask, green_mask
    
    def analyze_light_intensity(self, mask, original_image):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        masked_region = cv2.bitwise_and(original_image, original_image, mask=mask)
        mean_intensity = np.mean(masked_region[mask > 0]) if np.any(mask > 0) else 0
        
        return area, mean_intensity
    
    def detect_traffic_light_state(self, image_path, show_analysis=False):
        image = self.load_image(image_path)
        if image is None:
            return 'unknown', 0.0
        
        hsv_image = self.preprocess_image(image)
        red_mask, yellow_mask, green_mask = self.detect_color_regions(hsv_image)
        
        red_area, red_intensity = self.analyze_light_intensity(red_mask, image)
        yellow_area, yellow_intensity = self.analyze_light_intensity(yellow_mask, image)
        green_area, green_intensity = self.analyze_light_intensity(green_mask, image)
        
        red_score = red_area * red_intensity
        yellow_score = yellow_area * yellow_intensity
        green_score = green_area * green_intensity
        
        scores = {'red': red_score, 'yellow': yellow_score, 'green': green_score}
        max_color = max(scores, key=scores.get)
        max_score = scores[max_color]
        
        total_score = sum(scores.values())
        confidence = (max_score / total_score) if total_score > 0 else 0.0
        
        if max_score < 1000:
            return 'unknown', confidence
        
        return max_color, confidence
    


def main():
    detector = TrafficLightDetector()
    image_path = "traffic_light.jpg"
    
    try:
        state, confidence = detector.detect_traffic_light_state(image_path, show_analysis=True)
        
        print(f"Traffic Light State: {state.upper()}")
        print(f"Confidence: {confidence:.2f}")
        
        if state == 'unknown':
            print("Could not determine traffic light state.")
        
    except Exception as e:
        print(f"Error processing image: {e}")

def process_multiple_images(image_paths):
    detector = TrafficLightDetector()
    results = []
    
    for path in image_paths:
        try:
            state, confidence = detector.detect_traffic_light_state(path)
            results.append({
                'image': path,
                'state': state,
                'confidence': confidence
            })
        except Exception as e:
            results.append({
                'image': path,
                'state': 'error',
                'confidence': 0.0,
                'error': str(e)
            })
    
    return results

if __name__ == "__main__":
    main()