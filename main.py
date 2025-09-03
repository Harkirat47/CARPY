#!/usr/bin/env python3
# MAIN_CAM_WEATHER_YOLO.py — Weather → CAM → YOLO with safer defaults, baseline refresh, blur HUD, and toggles.
import os
import cv2
import time
import types
import numpy as np
from typing import List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────
# REQUIRED: import CameraMonitor exactly as-is from CAMFINALREAL
# ─────────────────────────────────────────────────────────────
from CAMFINALREAL import CameraMonitor  # unchanged

# ─────────────────────────────────────────────────────────────
# REQUIRED: import Weather model parts exactly from your file
# ─────────────────────────────────────────────────────────────
# WeatherModel.py must define: process_frame
from WeatherModel import process_frame

# Optional shim for externally loaded YOLO
try:
    import yolo_loaded  # exposes yolo_predict(rgb) or yolo_model
except Exception:
    yolo_loaded = None

# Video processing
LIMIT_RESOLUTION          = True    # True to downscale large videos, False to use original size
MAX_WIDTH                 = 640     # Max width for downscaled video
MAX_HEIGHT                = 480     # Max height for downscaled video

# ─────────────────────────────────────────────────────────────
# TUNABLE VALUES (EDIT NUMBERS/PATHS ONLY)
# ─────────────────────────────────────────────────────────────
# Video source: Set a path to process a file, or set to None to use a live camera.
VIDEO_FILE_PATH           = "test.mov"

# Camera & runner (used if VIDEO_FILE_PATH is None)
CAM_INDEX                 = 1
FALLBACK_INDEX            = 0
CAP_WIDTH                 = 640
CAP_HEIGHT                = 480
CAP_FPS                   = 30
INIT_MIN_SCORE            = -1.0    # -1 => auto via tracker.persistence_hi; else explicit 0..1
START_WITH_GRID           = 1       # 1 show grid, 0 hide

# CameraMonitor constructor (matches CAM.CameraMonitor.__init__)
GRID_X                    = 8
GRID_Y                    = 6
PERSISTENCE_HI            = 0.7    # safer (was 0.85)
RATIO_LAP                 = 0.60    # safer (was 0.55)
RATIO_EDGE                = 0.60    # safer (was 0.55)
RATIO_CONTR               = 0.9    # safer (was 0.65)
GLOBAL_MIN_EDGES          = 0.1
GLOBAL_MIN_LAP            = 10.0
FREEZE_MARGIN             = 0.8
SEQ_LEN                   = 30
PIXEL_MASK_DECAY          = 0.98
HEALTHY_DEACTIVATE        = 50

# Post-init tracker overrides
EWMA_ALPHA                = 0.2
DECAY_PIXEL               = 0.98
DECAY_TILE                = 0.95
DELTA_THRESH              = 20
HOT_VAL                   = 240
DEAD_VAL                  = 10
VAR_THRESH                = 2.0
MIN_TILE_AREA             = 400     # safer (was 400)

# Post-init corrector overrides
CORR_PIXEL_MASK_DECAY     = -1.0    # -1 keep ctor; else 0..1
CORR_HEALTHY_DEACTIVATE   = -1      # -1 keep ctor; else >=1

# Enhancement strength (CLAHE/unsharp) — set both to apply
CLAHE_CLIP_LIMIT          = -1.0    # keep corrector default (milder than forcing)
CLAHE_TILE_W              = -1
CLAHE_TILE_H              = -1

# Internal thresholds (monkey-patched)
PERSIST_MASK_BINARY_THR   = 0.70    # less sticky inpaint (was 0.40)
ACTIVE_TILE_HEALTHY_THR   = 0.15    # faster deactivation (was 0.15)

# HUD tweak (cosmetic)
GRID_COLOR_R              = 80
GRID_COLOR_G              = 80
GRID_COLOR_B              = 80
HUD_FONT_SCALE            = 0.65

# Weather model config
WEATHER_WEIGHTS_PATH      = "generator-3.pth"
WEATHER_INPUT_SIZE        = 256
WEATHER_MEDIAN_KERNEL     = 3

# YOLO config
YOLO_WEIGHTS_PATH         = "yolov8n.pt"  # used if auto-loading via Ultralytics
YOLO_CONF_THRESH          = 0.25
YOLO_IOU_THRESH           = 0.45
YOLO_EXPECTS_RGB          = True

# Baseline refresh policy
AUTO_BASELINE_REFRESH     = True     # automatically refresh baseline when enough logs exist
BASELINE_REFRESH_EVERY_S  = 120.0    # minimum seconds between auto-refresh attempts

# Window title
WINDOW_TITLE              = "Weather → Camera Blindspot Monitor → YOLO (Press h for help)"

# ─────────────────────────────────────────────────────────────
# Weather init & runner (wraps WeatherModel.process_frame)
# ─────────────────────────────────────────────────────────────
class WeatherRunner:
    """
    Wrapper for the classical CV-based dehazing algorithm in WeatherModel.py.
    This replaces the original PyTorch-based implementation.
    """
    def __init__(self):
        pass

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # The process_frame function from the provided WeatherModel.py is called directly.
        return process_frame(frame_bgr)

# ─────────────────────────────────────────────────────────────
# YOLO init & runner (external or auto-load)
# ─────────────────────────────────────────────────────────────
class YoloRunner:
    def __init__(self,
                 conf: float = YOLO_CONF_THRESH,
                 iou: float = YOLO_IOU_THRESH,
                 expects_rgb: bool = YOLO_EXPECTS_RGB,
                 weights_path: str = YOLO_WEIGHTS_PATH):
        self.conf = float(conf)
        self.iou  = float(iou)
        self.expects_rgb = bool(expects_rgb)
        self.model = None
        self.pred_fn = None

        # Preferred: external callable
        if yolo_loaded and hasattr(yolo_loaded, "yolo_predict") and callable(yolo_loaded.yolo_predict):
            self.pred_fn = yolo_loaded.yolo_predict
            return

        # Next: external Ultralytics-like model
        if yolo_loaded and hasattr(yolo_loaded, "yolo_model"):
            self.model = getattr(yolo_loaded, "yolo_model")
            return

        # Finally: local weights
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
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                return list(self.pred_fn(rgb))
            except Exception as e:
                print(f"[YOLO] External yolo_predict() failed: {e}")
                return []
        if self.model is None:
            return []
        img_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.expects_rgb else frame_bgr
        try:
            results = self.model.predict(img_in, conf=self.conf, iou=self.iou, verbose=False)
            if not results:
                return []
            res0 = results[0]
            dets: List[Dict[str, Any]] = []
            names = getattr(res0, "names", None) or getattr(self.model, "names", None) or {}
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
# Apply numeric overrides & internal thresholds
# ─────────────────────────────────────────────────────────────
def _apply_numeric_overrides(monitor: CameraMonitor):
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
# Draw helpers
# ─────────────────────────────────────────────────────────────
def draw_blindspots(frame: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        kind = b.get("kind", b.get("cause", "unknown"))
        why = b.get("why", {})
        issue = why.get("issue", "unknown")
        conf  = why.get("issue_conf", None)

        extras = []
        if "r_lap"     in why: extras.append(f"rl:{why['r_lap']:.2f}")
        if "r_edge"    in why: extras.append(f"re:{why['r_edge']:.2f}")
        if "r_contr"   in why: extras.append(f"rc:{why['r_contr']:.2f}")
        if "avg_hot"   in why: extras.append(f"h:{why['avg_hot']:.2f}")
        if "avg_dead"  in why: extras.append(f"d:{why['avg_dead']:.2f}")
        if "avg_stuck" in why: extras.append(f"s:{why['avg_stuck']:.2f}")
        if conf is not None:    extras.append(f"{issue[:3]}:{conf:.2f}")

        color = (0, 0, 255) if issue == "camera" else (0, 165, 255)
        label = f"{kind}" + (f" ({', '.join(extras[:3])})" if extras else "")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def draw_grid(frame: np.ndarray, grid: Tuple[int, int]=(8, 6), color: Tuple[int, int, int]=(80, 80, 80)) -> np.ndarray:
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
# Helpers that use CAMFINALREAL extras
# ─────────────────────────────────────────────────────────────
def wholeframe_blur_cause(monitor: CameraMonitor, frame_bgr: np.ndarray) -> str:
    """Use BlurDetector over the whole frame for a simple cause label."""
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(gray, dtype=bool)
        lap_var, cause = monitor.blur_detector.analyze_region(gray, mask)
        if cause:
            return str(cause)
        if lap_var is not None and lap_var < 20:
            return "very_blurry"
    except Exception:
        pass
    return ""

_last_baseline_refresh_t = 0.0

def maybe_refresh_baseline(monitor: CameraMonitor, force: bool = False) -> bool:
    """Refresh drift baseline using BaselineUpdater + DriftLogger history."""
    global _last_baseline_refresh_t
    now = time.time()
    if not force and (now - _last_baseline_refresh_t) < BASELINE_REFRESH_EVERY_S:
        return False
    try:
        new_base = monitor.updater.update_baseline_from_logs(monitor.logger)
        if new_base is not None:
            monitor.drift_detector.set_baseline(new_base)
            _last_baseline_refresh_t = now
            print("[Baseline] Refreshed from recent logs.")
            return True
    except Exception as e:
        print(f"[Baseline] Refresh failed: {e}")
    return False

def _is_strong_camera_smudge(b: Dict[str, Any]) -> bool:
    """Stricter gating for human actions (ack/persist/label) to avoid weak tiles."""
    why = b.get("why", {})
    if b.get("kind") != "smudge":
        return False
    if why.get("issue", "camera") != "camera":
        return False
    conf = float(why.get("issue_conf", 0.0))
    rlap = float(why.get("r_lap", 1.0))
    redg = float(why.get("r_edge", 1.0))
    rcon = float(why.get("r_contr", 1.0))
    # Strong either by model confidence or by very low relative texture/contrast
    return (conf >= 0.70) or (rlap < 0.45 and redg < 0.45 and rcon < 0.60)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # 0) Init Weather
    weather = None
    weather_enabled = True
    try:
        weather = WeatherRunner()
        print("[Weather] Dehazing model ready.")
    except Exception as e:
        print(f"[Weather] Disabled: {e}")
        weather_enabled = False
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

    # 4) Open video source (file or camera)
    cap = None
    if VIDEO_FILE_PATH:
        print(f"Opening video file: {VIDEO_FILE_PATH}")
        if not os.path.exists(VIDEO_FILE_PATH):
            print(f"Error: Video file not found at '{VIDEO_FILE_PATH}'")
            return
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    else:
        print(f"Opening camera index: {CAM_INDEX}")
        cap = cv2.VideoCapture(int(CAM_INDEX))
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(int(FALLBACK_INDEX))
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(CAP_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAP_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS,          int(CAP_FPS))

    if not cap or not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("🔍 Weather → Camera Blindspot + YOLO")
    print("   h=help | q=quit | a=ack/freeze | p=persist | s=label setting")
    print("   r=unfreeze recovered | u=deactivation pass | g=toggle grid")
    print("   [ / ] adjust min_score | w=toggle weather | v=toggle view | b=refresh baseline | c=toggle corrections")

    min_score = None if INIT_MIN_SCORE < 0 else float(INIT_MIN_SCORE)
    show_grid = bool(START_WITH_GRID)
    show_view_corrected_only = False  # v=toggle: False => annotated; True => YOLO input view
    corrections_enabled = True        # c=toggle

    # FPS meter
    t0 = time.time()
    frames = 0
    fps = 0.0

    # Baseline auto-refresh cadence
    last_auto_refresh = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file or failed to grab frame.")
            break

        if LIMIT_RESOLUTION:
            h, w = frame.shape[:2]
            if w > MAX_WIDTH or h > MAX_HEIGHT:
                ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
                new_dim = (int(w * ratio), int(h * ratio))
                frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

        # (1) WEATHER — optional and toggleable
        if weather is not None and weather_enabled:
            try:
                # process_frame from WeatherModel returns (corrected_image, haze_map)
                # We only need the corrected image for the next pipeline stage.
                enhanced, _ = weather(frame)
            except Exception as e:
                print(f"[Weather] Runtime error, disabling this session: {e}")
                enhanced = frame
                weather_enabled = False
        else:
            enhanced = frame

        # (2) CAM — update tracker and optionally apply corrections
        if corrections_enabled:
            corrected, meta = monitor.preprocess_for_yolo(enhanced, min_score=min_score)
        else:
            # Only update tracker; do NOT apply fixes
            monitor.blindspot_tracker.update(enhanced)
            # synthesize meta for HUD
            eff_min = (min_score if isinstance(min_score, float) else PERSISTENCE_HI)
            corrected = enhanced
            meta = {
                "boxes": monitor.blindspot_tracker.get_persistent_boxes(min_score=eff_min),
                "mask_nonzero": int(np.count_nonzero(monitor.blindspot_tracker.get_persistent_mask(min_score=eff_min))),
                "health": monitor.blindspot_tracker.camera_health(),
                "active_tiles": 0
            }

        # Drift signal (does NOT gate corrections) — use enhanced for analysis
        is_drift, distance, _ = monitor.process_frame(enhanced)

        # Optional: attempt auto baseline refresh (uses DriftLogger history)
        if AUTO_BASELINE_REFRESH and (time.time() - last_auto_refresh) >= BASELINE_REFRESH_EVERY_S:
            if maybe_refresh_baseline(monitor, force=False):
                last_auto_refresh = time.time()

        # (3) YOLO — run on corrected (what YOLO would actually see)
        dets = yolo.infer(corrected)
        yolo_out = yolo.draw(corrected, dets) if dets else corrected.copy()

        # Annotate display
        display_img = yolo_out if show_view_corrected_only else yolo_out.copy()

        # For hotkeys, use stricter camera-smudge selection to avoid persisting weak tiles
        raw_boxes = meta.get("boxes", [])
        strong_cam_boxes = [b for b in raw_boxes if _is_strong_camera_smudge(b)]

        if not show_view_corrected_only:
            if raw_boxes:
                display_img = draw_blindspots(display_img, raw_boxes)
            if show_grid:
                display_img = draw_grid(
                    display_img,
                    grid=(int(GRID_X), int(GRID_Y)),
                    color=(int(GRID_COLOR_B), int(GRID_COLOR_G), int(GRID_COLOR_R))
                )

        # HUD
        health = meta.get("health", {})
        ms_show = f"{min_score:.2f}" if isinstance(min_score, float) else "auto"
        blur_cause = wholeframe_blur_cause(monitor, enhanced)
        hud_bits = [
            ("Drift" if is_drift else "Stable"),
            f"d={distance:.2f}",
            f"maskNZ={meta.get('mask_nonzero',0)}",
            f"tiles={health.get('tiles_flagged',0)}",
            f"active={meta.get('active_tiles',0)}",
            f"min_score={ms_show}",
            f"fps={fps:.1f}",
            f"corr={'on' if corrections_enabled else 'off'}"
        ]
        if blur_cause:
            hud_bits.append(f"blur={blur_cause}")
        if weather is not None:
            hud_bits.append(f"wx={'on' if weather_enabled else 'off'}")
        color = (0, 0, 255) if is_drift else (0, 200, 0)
        cv2.putText(display_img, " | ".join(hud_bits), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, float(HUD_FONT_SCALE), color, 2)

        dets = yolo.infer(frame)
        yolo_out = yolo.draw(frame, dets) if dets else frame.copy()
        combined_display = np.hstack([yolo_out, display_img])
        cv2.imshow(WINDOW_TITLE, combined_display)

        # FPS calc
        frames += 1
        if frames % 15 == 0:
            dt = time.time() - t0
            fps = 15.0 / dt if dt > 0 else 0.0
            t0 = time.time()

        # Hotkeys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break

        

        elif key == ord('h'):
            print("Hotkeys:")
            print("  q=quit | a=ack/freeze | p=persist | s=label setting")
            print("  r=unfreeze recovered | u=deactivation pass | g=toggle grid")
            print("  [ / ] adjust min_score | w=toggle weather | v=toggle view | b=refresh baseline | c=toggle corrections")

        elif key == ord('a'):
            try:
                monitor.acknowledge_current_tiles(strong_cam_boxes)
                print(f"Frozen (ack) {len(strong_cam_boxes)} strong smudge tiles as camera (re-alert if ~15% worse).")
            except Exception as e:
                print(f"ack error: {e}")

        elif key == ord('p'):
            try:
                monitor.persist_camera_corrections(strong_cam_boxes)
                print(f"Persisted camera corrections + frozen tiles for {len(strong_cam_boxes)} strong tiles.")
            except Exception as e:
                print(f"persist error: {e}")

        elif key == ord('s'):
            try:
                monitor.label_current_as_setting([b for b in raw_boxes if b.get('kind')=='smudge'])
                print("Labeled current smudge tiles as setting issues (LSTM training samples added).")
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

        elif key == ord('w'):
            if weather is None:
                print("Weather not available.")
            else:
                weather_enabled = not weather_enabled
                print(f"weather → {'on' if weather_enabled else 'off'}")

        elif key == ord('v'):
            show_view_corrected_only = not show_view_corrected_only
            print(f"view → {'YOLO input only' if show_view_corrected_only else 'annotated'}")

        elif key == ord('b'):
            if maybe_refresh_baseline(monitor, force=True):
                print("Baseline refreshed from logs.")
            else:
                print("Baseline refresh skipped or not enough logs yet.")

        elif key == ord('c'):
            corrections_enabled = not corrections_enabled
            print(f"corrections → {'on' if corrections_enabled else 'off'}")

        elif key == ord('['):
            if min_score is None:
                min_score = 0.90  # start high in manual mode
            min_score = max(0.50, min_score - 0.05)
            print(f"min_score → {min_score:.2f}")

        elif key == ord(']'):
            if min_score is None:
                min_score = 0.90  # start high in manual mode
            min_score = min(0.99, min_score + 0.05)
            print(f"min_score → {min_score:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()