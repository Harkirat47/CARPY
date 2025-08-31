#!/usr/bin/env python3
# CAM_TEST.py — tune CameraMonitor by editing ONLY the numbers below.
import cv2
import numpy as np
import time
import types
from CAMFINALREAL import CameraMonitor  # your CAM.py (unchanged)

# ─────────────────────────────────────────────────────────────
# TUNABLE VALUES (EDIT NUMBERS ONLY)
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


def _apply_numeric_overrides(monitor):
    """Apply all constants above to the constructed monitor."""
    t = monitor.blindspot_tracker
    c = monitor.blindspot_corrector

    # Tracker numeric fields
    t.ewma_alpha   = float(EWMA_ALPHA)
    t.decay_pixel  = float(DECAY_PIXEL)
    t.decay_tile   = float(DECAY_TILE)
    t.delta_thresh = float(DELTA_THRESH)
    t.hot_val      = float(HOT_VAL)
    t.dead_val     = float(DEAD_VAL)
    t.var_thresh   = float(VAR_THRESH)
    t.min_tile_area= int(MIN_TILE_AREA)
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

def draw_blindspots(frame, boxes):
    """Draw persistent blindspots with short, readable labels."""
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        kind = b.get("kind", b.get("cause", "unknown"))
        why = b.get("why", {})
        issue = why.get("issue", "unknown")
        conf  = why.get("issue_conf", None)

        # compact extras
        extras = []
        if "r_lap"   in why: extras.append(f"rl:{why['r_lap']:.2f}")
        if "r_edge"  in why: extras.append(f"re:{why['r_edge']:.2f}")
        if "r_contr" in why: extras.append(f"rc:{why['r_contr']:.2f}")
        if "avg_hot" in why: extras.append(f"h:{why['avg_hot']:.2f}")
        if "avg_dead" in why: extras.append(f"d:{why['avg_dead']:.2f}")
        if "avg_stuck" in why: extras.append(f"s:{why['avg_stuck']:.2f}")
        if conf is not None:  extras.append(f"{issue[:3]}:{conf:.2f}")

        color = (0, 0, 255) if issue == "camera" else (0, 165, 255)
        label = f"{kind}" + (f" ({', '.join(extras[:3])})" if extras else "")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def draw_grid(frame, grid=(8,6), color=(80,80,80)):
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

def main():
    # 1) Build monitor with constructor tunables
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

    # 3) Camera open
    cap = cv2.VideoCapture(int(CAM_INDEX))
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(int(FALLBACK_INDEX))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(CAP_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAP_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS,          int(CAP_FPS))

    print("🔍 Camera Blindspot + YOLO preproc")
    print("   q=quit | a=ack/freeze | p=persist corrections | s=label setting | r=unfreeze recovered | u=auto-deactivate pass")
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

        # Correct for YOLO (updates tracker, applies pixel inpaint & persistent tile fixes)
        corrected, meta = monitor.preprocess_for_yolo(frame, min_score=min_score)

        # Optional drift signal (does NOT gate corrections)
        is_drift, distance, _boxes_unused = monitor.process_frame(frame)

        # Annotate corrected feed (what YOLO would see)
        annotated = corrected.copy()
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

        cv2.imshow("Camera Blindspot Monitor (corrected feed)", annotated)

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
