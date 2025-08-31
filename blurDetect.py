import cv2
import numpy as np

class BlurDetector:
    def __init__(self,
                 blur_thresh=30,
                 brightness_thresh=60,
                 contrast_thresh=20,
                 edge_thresh=0.05,
                 min_area=500):
        self.blur_thresh = blur_thresh
        self.brightness_thresh = brightness_thresh
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh
        self.min_area = min_area

    def analyze_region(self, gray, region_mask):
        region = gray[region_mask]
        if region.size == 0:
            return None, None

        lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
        brightness = np.mean(region)
        contrast = np.std(region)

        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / (region.size + 1e-5)

        cause = None
        if lap_var < self.blur_thresh:
            if brightness < self.brightness_thresh:
                cause = "low_light"
            elif contrast < self.contrast_thresh:
                cause = "low_contrast"
            elif edge_density < self.edge_thresh:
                cause = "low_edges"

        return lap_var, cause

    def detect_blindspots(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Define a grid of patches (e.g. 8x6)
        boxes = []
        step_x = w // 8
        step_y = h // 6

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                x_end = min(x + step_x, w)
                y_end = min(y + step_y, h)

                patch = gray[y:y_end, x:x_end]
                mask = np.s_[y:y_end, x:x_end]

                lap_var, cause = self.analyze_region(gray, mask)

                if cause:
                    box = {
                        "bbox": (x, y, x_end, y_end),
                        "cause": cause,
                        "lap_var": lap_var
                    }
                    boxes.append(box)

        return boxes
