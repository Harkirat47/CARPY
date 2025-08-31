import os
import time
import json
import cv2
from pathlib import Path
from collections import deque
from datetime import datetime

class DriftLogger:
    def __init__(self, log_dir="drift_logs", cooldown=5.0, max_history=6):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.cooldown = cooldown  # in seconds
        self.last_log_time = 0

        self.history = deque(maxlen=max_history)  # recent logs in memory

    def log(self, image, distance, blindspots):
        now = time.time()
        if now - self.last_log_time < self.cooldown:
            return False  # skip logging too frequently

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"drift_{timestamp}_d{distance:.2f}"

        # Save image
        img_path = self.log_dir / f"{filename_base}.jpg"
        cv2.imwrite(str(img_path), image)

        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "distance": distance,
            "blindspots": blindspots  # list of {"bbox", "cause", "lap_var"}
        }

        # Save metadata
        meta_path = self.log_dir / f"{filename_base}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Keep in memory
        self.history.append({
            "img_path": str(img_path),
            "meta_path": str(meta_path),
            "features": None  # to be filled in by baseline updater
        })

        self.last_log_time = now
        print(f"⚠️ Logged drift to {img_path.name} with {len(blindspots)} blindspots.")
        return True

    def get_recent_logs(self):
        return list(self.history)
