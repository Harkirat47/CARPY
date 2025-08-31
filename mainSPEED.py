#!/usr/bin/env python3
# FAST_MAIN.py — Weather → CAM → YOLO (streamlined)

import os, cv2, time, torch
import numpy as np
from torchvision import transforms

# ---- imports from your own files ----
from CAMFINALREAL import CameraMonitor
from WeatherModel import Generator, MedianFilterTransform, process_frame

# ---- CONFIG (your overrides) ----
CAM_INDEX, FALLBACK_INDEX = 1, 0
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 640, 480, 30
GRID_X, GRID_Y = 4, 3
PERSISTENCE_HI, RATIO_LAP, RATIO_EDGE, RATIO_CONTR = 0.85, 0.55, 0.55, 0.65
GLOBAL_MIN_EDGES, GLOBAL_MIN_LAP = 0.01, 30.0
FREEZE_MARGIN, SEQ_LEN = 0.85, 30
PIXEL_MASK_DECAY, HEALTHY_DEACTIVATE = 0.98, 30
WEATHER_WEIGHTS_PATH, WEATHER_INPUT_SIZE, WEATHER_MEDIAN_KERNEL = "generator.pth", 256, 3
YOLO_WEIGHTS_PATH, YOLO_CONF, YOLO_IOU = "yolo12s.pt", 0.25, 0.45
WINDOW_TITLE = "Weather → CAM → YOLO"

# ---- Weather ----
class WeatherRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Generator().to(self.device).eval()
        self.model.load_state_dict(torch.load(WEATHER_WEIGHTS_PATH, map_location=self.device))
        self.median = MedianFilterTransform(kernel_size=WEATHER_MEDIAN_KERNEL)
        self.transform = transforms.Compose([
            transforms.Resize((WEATHER_INPUT_SIZE, WEATHER_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        out = process_frame(self.model, frame_bgr, self.device, self.median, self.transform)
        return cv2.resize(out, (w, h)) if out.shape[:2] != (h, w) else out

# ---- YOLO ----
class YoloRunner:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(YOLO_WEIGHTS_PATH)
    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(rgb, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
        out = frame_bgr.copy()
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            x1,y1,x2,y2 = map(int, box)
            label = f"{res.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        return out

# ---- MAIN ----
def main():
    weather = WeatherRunner()
    yolo = YoloRunner()
    monitor = CameraMonitor(
        grid=(GRID_X, GRID_Y), persistence_hi=PERSISTENCE_HI,
        ratio_lap=RATIO_LAP, ratio_edge=RATIO_EDGE, ratio_contr=RATIO_CONTR,
        global_min_edges=GLOBAL_MIN_EDGES, global_min_lap=GLOBAL_MIN_LAP,
        freeze_margin=FREEZE_MARGIN, seq_len=SEQ_LEN,
        pixel_mask_decay=PIXEL_MASK_DECAY,
        healthy_frames_to_deactivate=HEALTHY_DEACTIVATE
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): cap = cv2.VideoCapture(FALLBACK_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        f1 = weather(frame)                       # Weather
        corrected, meta = monitor.preprocess_for_yolo(f1)  # CAM
        f2 = yolo(corrected)                      # YOLO

        fps = frames / (time.time()-t0+1e-6); frames+=1
        cv2.putText(f2, f"FPS:{fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2)

        cv2.imshow(WINDOW_TITLE, f2)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
