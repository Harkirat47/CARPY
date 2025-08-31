#!/usr/bin/env python3
# MAIN_CAM_WEATHER_YOLO.py — merges Weather → CAM → YOLO (in that order), preserving your CAM_TEST behavior.
import os
import cv2
import time
import types
import torch
import numpy as np
from typing import List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────
# REQUIRED: import CameraMonitor exactly as-is from CAMFINALREAL
# ─────────────────────────────────────────────────────────────
from CAMFINALREAL import CameraMonitor  # unchanged

# ─────────────────────────────────────────────────────────────
# REQUIRED: import Weather model parts exactly from your file
# ─────────────────────────────────────────────────────────────
# weather_model.py must define: Generator, MedianFilterTransform, process_frame
from torchvision import transforms
from WeatherModel import Generator, MedianFilterTransform, process_frame

# Optional: If your YOLO is already loaded elsewhere, you may provide a simple module:
#   # yolo_loaded.py
#   # expose either `yolo_model` (Ultralytics-like) or a callable `yolo_predict(frame_rgb)->list[dict]`
#   from ultralytics import YOLO
#   yolo_model = YOLO("yolo12s.pt")
#
# This main will first try `yolo_loaded.yolo_predict`, then `yolo_loaded.yolo_model`,
# else it will try to load "yolo12s.pt" itself if present.
try:
    import yolo_loaded  # optional convenience shim
except Exception:
    yolo_loaded = None

# ─────────────────────────────────────────────────────────────
# TUNABLE VALUES (EDIT NUMBERS/PATHS ONLY)
# ─────────────────────────────────────────────────────────────
# Camera & runner
CAM_INDEX                 = 1       # fallback used if this fails
FALLBACK_INDEX            = 0
CAP_WIDTH                 = 640
CAP_HEIGHT                = 480
CAP_FPS                   = 30
INIT_MIN_SCORE            = -1.0    # -1 => auto (use tracker.persistence_hi), else 0.00..1.00
START_WITH_GRID           = 1       # 1 => show grid, 0 => hide

# CameraMonitor constructor (top-level, matches CAM.CameraMonitor.__init__)
GRID_X                    = 8
GRID_Y                    = 6
PERSISTENCE_HI            = 0.85
RATIO_LAP                 = 0.55
RATIO_EDGE                = 0.55
RATIO_CONTR               = 0.65
GLOBAL_MIN_EDGES          = 0.01
GLOBAL_MIN_LAP            = 30.0
FREEZE_MARGIN             = 0.85
SEQ_LEN                   = 30
PIXEL_MASK_DECAY          = 0.98
HEALTHY_DEACTIVATE        = 30

# Post-init tracker overrides (BlindspotTracker fields not in the ctor)
EWMA_ALPHA                = 0.05
DECAY_PIXEL               = 0.98
DECAY_TILE                = 0.95
DELTA_THRESH              = 20
HOT_VAL                   = 245
DEAD_VAL                  = 10
VAR_THRESH                = 2.0
MIN_TILE_AREA             = 400
# (You can redo FREEZE_MARGIN here by changing FREEZE_MARGIN above; we pass it in ctor)

# Post-init corrector overrides (BlindspotCorrector)
CORR_PIXEL_MASK_DECAY     = -1.0    # -1 => keep ctor value; else 0..1
CORR_HEALTHY_DEACTIVATE   = -1      # -1 => keep ctor value; else >=1

# Enhancement strength (CLAHE/unsharp) — set both to apply
CLAHE_CLIP_LIMIT          = -1.0    # -1 => keep default; else >0 (e.g., 2.5)
CLAHE_TILE_W              = -1      # -1 => keep default
CLAHE_TILE_H              = -1      # -1 => keep default

# Internal thresholds (monkey-patched)
PERSIST_MASK_BINARY_THR   = 0.40    # default in CAM is 0.40; lower = stickier pixel mask
ACTIVE_TILE_HEALTHY_THR   = 0.15    # default in CAM is 0.15; lower = easier to count as healthy

# HUD tweak (cosmetic)
GRID_COLOR_R              = 80
GRID_COLOR_G              = 80
GRID_COLOR_B              = 80
HUD_FONT_SCALE            = 0.65

# Weather model config
WEATHER_WEIGHTS_PATH      = "generator.pth"
WEATHER_INPUT_SIZE        = 512
WEATHER_MEDIAN_KERNEL     = 3

# YOLO config
YOLO_WEIGHTS_PATH         = "yolo12s.pt"  # used if auto-loading via Ultralytics
YOLO_CONF_THRESH          = 0.25
YOLO_IOU_THRESH           = 0.45
YOLO_EXPECTS_RGB          = True         # most Ultralytics models expect RGB

# Window title
WINDOW_TITLE              = "Weather → Camera Blindspot Monitor → YOLO (Press q to quit)"


# ─────────────────────────────────────────────────────────────
# Weather init & runner (wraps your weather_model.process_frame)
# ─────────────────────────────────────────────────────────────
class WeatherRunner:
    def __init__(self,
                 weights_path: str = WEATHER_WEIGHTS_PATH,
                 input_size: int = WEATHER_INPUT_SIZE,
                 median_kernel: int = WEATHER_MEDIAN_KERNEL,
                 device: torch.device | None = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = Generator().to(self.device)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weather model weights not found at '{weights_path}'. "
                f"Place your trained weights there or update WEATHER_WEIGHTS_PATH."
            )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.median_filter = MedianFilterTransform(kernel_size=int(median_kernel))
        self.transform = transforms.Compose([
            transforms.Resize((int(input_size), int(input_size))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns enhanced BGR image resized back to original WxH."""
        h, w = frame_bgr.shape[:2]
        with torch.no_grad():
            out_bgr = process_frame(self.model, frame_bgr, self.device, self.median_filter, self.transform)
        if out_bgr.shape[:2] != (h, w):
            out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return out_bgr


# ─────────────────────────────────────────────────────────────
# YOLO init & runner (works with already-loaded or auto-loaded)
# ─────────────────────────────────────────────────────────────
class YoloRunner:
    """
    Attempts the following, in order:
    1) Use yolo_loaded.yolo_predict(frame_rgb) if provided (returns list of det dicts).
    2) Use yolo_loaded.yolo_model (Ultralytics-like): model(img)[0] → parse boxes.
    3) If YOLO_WEIGHTS_PATH exists, try to load ultralytics.YOLO and parse.

    Detection dict format (internal):
        {'xyxy': (x1, y1, x2, y2), 'conf': float, 'cls': int, 'name': str}
    """
    def __init__(self,
                 conf: float = YOLO_CONF_THRESH,
                 iou: float = YOLO_IOU_THRESH,
                 expects_rgb: bool = YOLO_EXPECTS_RGB,
                 weights_path: str = YOLO_WEIGHTS_PATH):
        self.conf = float(conf)
        self.iou  = float(iou)
        self.expects_rgb = bool(expects_rgb)
        self.model = None
        self.pred_fn = None  # external callable: pred_fn(frame_rgb) -> list[dict]

        # Preferred: external callable
        if yolo_loaded and hasattr(yolo_loaded, "yolo_predict") and callable(yolo_loaded.yolo_predict):
            self.pred_fn = yolo_loaded.yolo_predict
            return

        # Next: external Ultralytics-like model object
        if yolo_loaded and hasattr(yolo_loaded, "yolo_model"):
            self.model = getattr(yolo_loaded, "yolo_model")
            return

        # Finally: try to load locally if weights exist
        if os.path.exists(weights_path):
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(weights_path)
            except Exception as e:
                print(f"[YOLO] Could not import/load ultralytics YOLO: {e}")
                self.model = None
        else:
            print(f"[YOLO] No external model and weights not found at '{weights_path}'. Running without detections.")

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.pred_fn is not None:
            # External callable expects RGB by convention
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                return list(self.pred_fn(rgb))
            except Exception as e:
                print(f"[YOLO] External yolo_predict() failed: {e}")
                return []

        if self.model is None:
            return []

        # Assume Ultralytics-like model
        img_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.expects_rgb else frame_bgr
        try:
            results = self.model.predict(img_in, conf=self.conf, iou=self.iou, verbose=False)
            if not results:
                return []
            res0 = results[0]
            dets: List[Dict[str, Any]] = []
            names = getattr(res0, "names", None) or getattr(self.model, "names", None) or {}
            # Handle either .boxes.xyxy / .boxes.conf / .boxes.cls style
            if hasattr(res0, "boxes") and res0.boxes is not None:
                try:
                    xyxy = res0.boxes.xyxy.cpu().numpy()
                    conf = res0.boxes.conf.cpu().numpy()
                    cls  = res0.boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(float, xyxy[i])
                        c = float(conf[i]) if i < len(conf) else 0.0
                        k = int(cls[i]) if i < len(cls) else -1
                        name = names.get(k, str(k))
                        dets.append({'xyxy': (x1, y1, x2, y2), 'conf': c, 'cls': k, 'name': name})
                except Exception:
                    pass
            return dets
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            return []

    def draw(self, frame_bgr: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
        out = frame_bgr.copy()
        for d in dets:
            (x1, y1, x2, y2) = map(int, d.get('xyxy', (0, 0, 0, 0)))
            conf = float(d.get('conf', 0.0))
            name = str(d.get('name', d.get('cls', '?')))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1 + 2, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out


# ─────────────────────────────────────────────────────────────
# Apply numeric overrides & internal thresholds (unchanged logic)
# ─────────────────────────────────────────────────────────────
def _apply_numeric_overrides(monitor: CameraMonitor):
    """Apply all constants above to the constructed monitor."""
    t = monitor.blindspot_tracker
    c = monitor.blindspot_corrector

    # Tracker numeric fields
    t.ewma_alpha    = float(EWMA_ALPHA)
    t.decay_pixel   = float(DECAY_PIXEL)
    t.decay_tile    = float(DECAY_TILE)
    t.delta_thresh  = float(DELTA_THRESH)
    t.hot_val       = float(HOT_VAL)
    t.dead_val      = float(DEAD_VAL)
    t.var_thresh    = float(VAR_THRESH)
    t.min_tile_area = int(MIN_TILE_AREA)
    # freeze_margin handled by ctor via FREEZE_MARGIN

    # Corrector numeric fields (optional overrides)
    if CORR_PIXEL_MASK_DECAY >= 0:
        c.pixel_mask_decay = float(CORR_PIXEL_MASK_DECAY)
    if CORR_HEALTHY_DEACTIVATE >= 1:
        c.healthy_frames_to_deactivate = int(CORR_HEALTHY_DEACTIVATE)

    # CLAHE replacement if both dims are positive OR clip limit set
    if CLAHE_CLIP_LIMIT > 0 or (CLAHE_TILE_W > 0 and CLAHE_TILE_H > 0):
        clip  = float(CLAHE_CLIP_LIMIT) if CLAHE_CLIP_LIMIT > 0 else 2.0
        tiles = (int(CLAHE_TILE_W) if CLAHE_TILE_W > 0 else 8,
                 int(CLAHE_TILE_H) if CLAHE_TILE_H > 0 else 8)
        c._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)

    # Monkey patch: persistent mask binarization threshold
    thr = float(PERSIST_MASK_BINARY_THR)
    def _patched_update_mask(self, new_mask_uint8, _th=thr):
        if new_mask_uint8 is None:
            return np.zeros((1, 1), dtype=np.uint8)
        m = (new_mask_uint8 > 0).astype(np.float32)
        if self._persist_mask_float is None or self._persist_mask_float.shape != m.shape:
            self._persist_mask_float = m.copy()
        else:
            self._persist_mask_float = np.maximum(self._persist_mask_float * self.pixel_mask_decay, m)
        return (self._persist_mask_float >= _th).astype(np.uint8) * 255
    c._update_persistent_pixel_mask = types.MethodType(_patched_update_mask, c)

    # Monkey patch: active-tile healthy score threshold
    healthy_thr = float(ACTIVE_TILE_HEALTHY_THR)
    def _patched_deactivate(self, _th=healthy_thr):
        gx, gy = self.tracker.grid
        scores = self.tracker.tile_scores if self.tracker.tile_scores is not None else np.zeros((gy, gx))
        newly_deactivated = []
        for (j, i) in list(self.active_tiles):
            if scores[j, i] < _th:
                self.tile_healthy_count[j, i] += 1
            else:
                self.tile_healthy_count[j, i] = 0
            if self.tile_healthy_count[j, i] >= self.healthy_frames_to_deactivate:
                self.active_tiles.remove((j, i))
                newly_deactivated.append((j, i))
        return newly_deactivated
    c.deactivate_recovered_tiles = types.MethodType(_patched_deactivate, c)


# ─────────────────────────────────────────────────────────────
# Draw helpers (unchanged behavior)
# ─────────────────────────────────────────────────────────────
def draw_blindspots(frame: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
    """Draw persistent blindspots with short, readable labels."""
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        kind = b.get("kind", b.get("cause", "unknown"))
        why = b.get("why", {})
        issue = why.get("issue", "unknown")
        conf  = why.get("issue_conf", None)

        # compact extras
        extras = []
        if "r_lap"    in why: extras.append(f"rl:{why['r_lap']:.2f}")
        if "r_edge"   in why: extras.append(f"re:{why['r_edge']:.2f}")
        if "r_contr"  in why: extras.append(f"rc:{why['r_contr']:.2f}")
        if "avg_hot"  in why: extras.append(f"h:{why['avg_hot']:.2f}")
        if "avg_dead" in why: extras.append(f"d:{why['avg_dead']:.2f}")
        if "avg_stuck" in why: extras.append(f"s:{why['avg_stuck']:.2f}")
        if conf is not None:   extras.append(f"{issue[:3]}:{conf:.2f}")

        color = (0, 0, 255) if issue == "camera" else (0, 165, 255)
        label = f"{kind}" + (f" ({', '.join(extras[:3])})" if extras else "")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def draw_grid(frame: np.ndarray, grid: Tuple[int, int]=(8, 6), color: Tuple[int, int, int]=(80, 80, 80)) -> np.ndarray:
    """Sensor-aligned tile grid overlay for quick visual alignment."""
    h, w = frame.shape[:2]
    gx, gy = grid
    step_x = max(1, w // gx)
    step_y = max(1, h // gy)
    for i in range(1, gx):
        x = i * step_x
        cv2.line(frame, (x, 0), (x, h), color, 1)
    for j in range(1, gy):
        y = j * step_y
        cv2.line(frame, (0, y), (w, y), color, 1)
    return frame


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # 0) Init Weather
    try:
        weather = WeatherRunner(
            weights_path=WEATHER_WEIGHTS_PATH,
            input_size=WEATHER_INPUT_SIZE,
            median_kernel=WEATHER_MEDIAN_KERNEL
        )
        print(f"[Weather] Loaded weights from {WEATHER_WEIGHTS_PATH}")
    except Exception as e:
        print(f"[Weather] Disabled: {e}")
        weather = None

    # 1) Build CameraMonitor with constructor tunables
    monitor = CameraMonitor(
        grid=(int(GRID_X), int(GRID_Y)),
        persistence_hi=float(PERSISTENCE_HI),
        ratio_lap=float(RATIO_LAP),
        ratio_edge=float(RATIO_EDGE),
        ratio_contr=float(RATIO_CONTR),
        global_min_edges=float(GLOBAL_MIN_EDGES),
        global_min_lap=float(GLOBAL_MIN_LAP),
        freeze_margin=float(FREEZE_MARGIN),
        seq_len=int(SEQ_LEN),
        pixel_mask_decay=float(PIXEL_MASK_DECAY),
        healthy_frames_to_deactivate=int(HEALTHY_DEACTIVATE)
    )

    # 2) Apply numeric overrides & internal thresholds
    _apply_numeric_overrides(monitor)

    # 3) YOLO init
    yolo = YoloRunner(conf=YOLO_CONF_THRESH, iou=YOLO_IOU_THRESH, expects_rgb=YOLO_EXPECTS_RGB)
    if yolo.model is not None or yolo.pred_fn is not None:
        print("[YOLO] Ready.")
    else:
        print("[YOLO] Not available; running without detections.")

    # 4) Camera open
    cap = cv2.VideoCapture(int(CAM_INDEX))
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(int(FALLBACK_INDEX))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(CAP_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAP_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS,          int(CAP_FPS))

    print("🔍 Weather → Camera Blindspot + YOLO")
    print("   q=quit | a=ack/freeze | p=persist corrections | s=label setting | r=unfreeze recovered | u=deactivate pass")
    print("   g=toggle grid | [ / ] adjust min_score")

    min_score = None if INIT_MIN_SCORE < 0 else float(INIT_MIN_SCORE)
    show_grid = bool(START_WITH_GRID)

    # FPS meter
    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # (1) WEATHER — optional
        enhanced = weather(frame) if weather is not None else frame

        # (2) CAM preprocess_for_yolo (updates tracker, applies pixel inpaint & persistent tile fixes)
        corrected, meta = monitor.preprocess_for_yolo(enhanced, min_score=min_score)

        # Optional drift signal (does NOT gate corrections) — use enhanced for analysis
        is_drift, distance, _boxes_unused = monitor.process_frame(enhanced)

        # (3) YOLO — run on corrected (what YOLO would actually see)
        dets = yolo.infer(corrected)
        yolo_out = yolo.draw(corrected, dets) if dets else corrected.copy()

        # Annotate corrected/yolo feed with blindspots & grid
        annotated = yolo_out
        boxes = meta.get("boxes", [])
        if boxes:
            annotated = draw_blindspots(annotated, boxes)
        if show_grid:
            annotated = draw_grid(
                annotated,
                grid=(int(GRID_X), int(GRID_Y)),
                color=(int(GRID_COLOR_B), int(GRID_COLOR_G), int(GRID_COLOR_R))
            )

        # HUD
        health = meta.get("health", {})
        ms_show = f"{min_score:.2f}" if isinstance(min_score, float) else "auto"
        status = [
            "Drift" if is_drift else " Stable",
            f"d={distance:.2f}",
            f"maskNZ={meta.get('mask_nonzero',0)}",
            f"tiles={health.get('tiles_flagged',0)}",
            f"active={meta.get('active_tiles',0)}",
            f"min_score={ms_show}",
            f"fps={fps:.1f}"
        ]
        color = (0, 0, 255) if is_drift else (0, 200, 0)
        cv2.putText(annotated, " | ".join(status), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, float(HUD_FONT_SCALE), color, 2)

        cv2.imshow(WINDOW_TITLE, annotated)

        # FPS calc
        frames += 1
        if frames % 15 == 0:
            dt = time.time() - t0
            fps = 15.0 / dt if dt > 0 else 0.0
            t0 = time.time()

        # Hotkeys (preserved)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break

        elif key == ord('a'):
            try:
                monitor.acknowledge_current_tiles(boxes)
                print("Frozen (ack) current smudge tiles as camera. Re-alert only if ~15% worse.")
            except Exception as e:
                print(f"ack error: {e}")

        elif key == ord('p'):
            try:
                monitor.persist_camera_corrections(boxes)
                print("Persisted camera corrections + frozen tiles.")
            except Exception as e:
                print(f"persist error: {e}")

        elif key == ord('s'):
            try:
                monitor.label_current_as_setting(boxes)
                print("Labeled current tiles as setting issues.")
            except Exception as e:
                print(f"label error: {e}")

        elif key == ord('r'):
            monitor.maybe_unfreeze_recovered(strong=True)
            print("Unfroze tiles that look recovered (conservative).")

        elif key == ord('u'):
            monitor.unstick_recovered()
            print("Deactivation pass executed (recovered tiles removed from active set when eligible).")

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"grid → {'on' if show_grid else 'off'}")

        elif key == ord('['):
            if min_score is None:
                min_score = 0.85
            min_score = max(0.50, min_score - 0.05)
            print(f"min_score → {min_score:.2f}")

        elif key == ord(']'):
            if min_score is None:
                min_score = 0.85
            min_score = min(0.99, min_score + 0.05)
            print(f"min_score → {min_score:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
