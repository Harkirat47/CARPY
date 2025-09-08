#!/usr/bin/env python3
"""
Render the FULL video using a configuration taken from an optimizer CSV.

- Picks the best trial (max "score") by default, or a specific --trial.
- Runs YOLO on every frame (no skip), with the same preprocessing/corrections
  as your optimizer's render_best().
- Writes a side-by-side mp4: [enhanced-left | corrected-right].

Usage:
  python render_full_from_csv.py --video path/to/video.mp4 \
      --csv results_opt/optimizer_log.csv \
      --weights yolov8n.pt \
      --out results_opt/full_best.mp4 \
      --classes "car,truck,bus,motorcycle,bicycle,person"
  # OR to force a specific trial from the CSV:
  python render_full_from_csv.py --video ... --csv ... --trial 37

Assumes CAMFINALREAL.CameraMonitor and (optionally) WeatherModel.process_frame
are available, identical to your optimizer.
"""

import os, csv, json, argparse
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Import the same components your optimizer uses
from CAMFINALREAL import CameraMonitor  # unchanged

# Weather (optional, same behavior as optimizer)
try:
    from WeatherModel import process_frame as weather_process_frame
    HAVE_WEATHER = True
except Exception as e:
    HAVE_WEATHER = False
    weather_process_frame = None

# Optional shim for externally loaded YOLO
try:
    import yolo_loaded  # exposes yolo_predict(rgb) or yolo_model
except Exception:
    yolo_loaded = None

# --------------------------
# YOLO runner (copied from optimizer)
# --------------------------
class YoloRunner:
    def __init__(self, weights_path="yolov8n.pt"):
        import os
        self.model = None
        self.pred_fn = None
        self.device = "cpu"
        self.half = False

        if yolo_loaded and hasattr(yolo_loaded, "yolo_predict") and callable(yolo_loaded.yolo_predict):
            self.pred_fn = yolo_loaded.yolo_predict
            return
        if yolo_loaded and hasattr(yolo_loaded, "yolo_model"):
            self.model = getattr(yolo_loaded, "yolo_model")
            return

        if os.path.exists(weights_path):
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(weights_path)
                try:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.half = (self.device == "cuda")
                except Exception:
                    self.device = "cpu"
                    self.half = False
                print(f"[YOLO] loaded '{weights_path}' on {self.device} (half={self.half})")
            except Exception as e:
                print(f"[YOLO] Could not import/load ultralytics YOLO: {e}")
                self.model = None
        else:
            print(f"[YOLO] Weights not found at '{weights_path}'. Running without detections.")

    def infer(self, frame_bgr: np.ndarray, conf: float, iou: float, expects_rgb: bool=True) -> List[Dict[str, Any]]:
        if self.pred_fn is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                return list(self.pred_fn(rgb))
            except Exception as e:
                print(f"[YOLO] External yolo_predict() failed: {e}")
                return []
        if self.model is None:
            return []
        img_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if expects_rgb else frame_bgr
        try:
            results = self.model.predict(
                img_in, conf=float(conf), iou=float(iou),
                verbose=False, device=self.device, half=self.half
            )
            if not results:
                return []
            res0 = results[0]
            dets: List[Dict[str, Any]] = []
            names = getattr(res0, "names", None) or getattr(self.model, "names", None) or {}
            if hasattr(res0, "boxes") and res0.boxes is not None:
                xyxy = res0.boxes.xyxy.cpu().numpy()
                confs = res0.boxes.conf.cpu().numpy()
                cls  = res0.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    c = float(confs[i]) if i < len(confs) else 0.0
                    k = int(cls[i]) if i < len(cls) else -1
                    name = names.get(k, str(k))
                    dets.append({'xyxy': (x1, y1, x2, y2), 'conf': c, 'cls': k, 'name': name})
            return dets
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            return []

    @staticmethod
    def draw(frame_bgr: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
        out = frame_bgr.copy()
        for d in dets:
            (x1, y1, x2, y2) = map(int, d.get('xyxy', (0, 0, 0, 0)))
            conf = float(d.get('conf', 0.0))
            name = str(d.get('name', d.get('cls', '?')))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1 + 2, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out

# --------------------------
# Weather wrapper (FIXED)
# --------------------------
class WeatherRunner:
    def __init__(self, enabled=True):
        self.enabled = HAVE_WEATHER and enabled
        if self.enabled:
            print("[Weather] Dehazing is enabled.")
        else:
            print("[Weather] Dehazing is disabled or unavailable.")

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame_bgr
        try:
            # Call the weather processing function - it returns (dehazed_frame, transmission_map)
            result = weather_process_frame(frame_bgr)
            
            # The WeatherModel.process_frame returns a tuple: (corrected_frame, transmission_map)
            if isinstance(result, tuple) and len(result) == 2:
                enhanced = result[0]  # Take the dehazed frame
            else:
                # Fallback in case the function changes or returns something unexpected
                enhanced = result
            
            # Ensure the enhanced frame is valid
            if enhanced is not None and enhanced.shape == frame_bgr.shape:
                return enhanced.astype(np.uint8)
            else:
                print("[Weather] Invalid enhanced frame, returning original")
                return frame_bgr
                
        except Exception as e:
            # Print the error and disable for the rest of the session
            print(f"[Weather] Error during processing, disabling for this session: {e}")
            self.enabled = False
            return frame_bgr

# --------------------------
# Draw grid (copied)
# --------------------------
def draw_grid(frame: np.ndarray, grid=(8,6), color=(80,80,80)) -> np.ndarray:
    h, w = frame.shape[:2]
    gx, gy = grid
    step_x = max(1, w // gx); step_y = max(1, h // gy)
    out = frame.copy()
    for i in range(1, gx):
        x = i * step_x
        cv2.line(out, (x, 0), (x, h), color, 1)
    for j in range(1, gy):
        y = j * step_y
        cv2.line(out, (0, y), (w, y), color, 1)
    return out

# --------------------------
# Build monitor + numeric overrides (same knobs as optimizer)
# --------------------------
import types
def _apply_numeric_overrides(monitor: CameraMonitor, knobs: Dict[str, Any]):
    t = monitor.blindspot_tracker
    c = monitor.blindspot_corrector
    t.ewma_alpha    = float(knobs["ewma_alpha"])
    t.decay_pixel   = float(knobs["decay_pixel"])
    t.decay_tile    = float(knobs["decay_tile"])
    t.delta_thresh  = float(knobs["delta_thresh"])
    t.hot_val       = float(knobs["hot_val"])
    t.dead_val      = float(knobs["dead_val"])
    t.var_thresh    = float(knobs["var_thresh"])
    t.min_tile_area = int(knobs["min_tile_area"])

    clip = float(knobs["clahe_clip"])
    tw, th = int(knobs["clahe_tile_w"]), int(knobs["clahe_tile_h"])
    if clip > 0 or (tw > 0 and th > 0):
        clip_final  = clip if clip > 0 else 2.0
        grid_final  = (tw if tw > 0 else 8, th if th > 0 else 8)
        c._clahe = cv2.createCLAHE(clipLimit=clip_final, tileGridSize=grid_final)

    thr = float(knobs["persist_mask_thr"])
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

    healthy_thr = float(knobs["active_healthy_thr"])
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

def build_monitor(kn: Dict[str, Any]) -> CameraMonitor:
    mon = CameraMonitor(
        grid=(int(kn["grid_x"]), int(kn["grid_y"])),
        persistence_hi=float(kn["persistence_hi"]),
        ratio_lap=float(kn["ratio_lap"]),
        ratio_edge=float(kn["ratio_edge"]),
        ratio_contr=float(kn["ratio_contr"]),
        global_min_edges=float(kn["global_min_edges"]),
        global_min_lap=float(kn["global_min_lap"]),
        freeze_margin=float(kn["freeze_margin"]),
        seq_len=int(kn["seq_len"]),
        pixel_mask_decay=float(kn["pixel_mask_decay"]),
        healthy_frames_to_deactivate=int(kn["healthy_deactivate"])
    )
    _apply_numeric_overrides(mon, kn)
    return mon

# --------------------------
# CSV helpers
# --------------------------
# Which columns we expect & their types (for robust coercion)
_INT_KEYS = {
    "limit_resolution", "start_with_grid", "weather_on", "grid_x", "grid_y",
    "healthy_deactivate", "min_tile_area", "clahe_tile_w", "clahe_tile_h", "seq_len",
    "trial", "auto_baseline"
}
_FLOAT_KEYS = {
    "yolo_conf", "yolo_iou", "persistence_hi", "ratio_lap", "ratio_edge", "ratio_contr",
    "global_min_edges", "global_min_lap", "freeze_margin",
    "pixel_mask_decay", "ewma_alpha", "decay_pixel", "decay_tile",
    "delta_thresh", "hot_val", "dead_val", "var_thresh",
    "persist_mask_thr", "active_healthy_thr", "baseline_every_s", "init_min_score",
    "score", "avg_new", "avg_conf_gain", "avg_mask_nz", "avg_active", "fps"
}

def _coerce_row(d: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None or v == "":
            continue
        if k in _INT_KEYS:
            try: out[k] = int(float(v))
            except: out[k] = 0
        elif k in _FLOAT_KEYS:
            try: out[k] = float(v)
            except: out[k] = 0.0
        else:
            # keep as-is for any unknown or stringy field
            out[k] = v
    # sensible defaults if missing
    out.setdefault("grid_x", 8); out.setdefault("grid_y", 6)
    out.setdefault("limit_resolution", 0)
    out.setdefault("start_with_grid", 0)
    out.setdefault("weather_on", 1)
    out.setdefault("yolo_conf", 0.25)
    out.setdefault("yolo_iou", 0.5)
    out.setdefault("init_min_score", -1.0)
    return out

def load_best_or_trial(csv_path: str, trial: Optional[int]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_coerce_row(r))
    if not rows:
        raise RuntimeError("CSV is empty or unreadable.")

    oks = [r for r in rows if r.get("ok", "True") in (True, "True", "true", "1")]
    if not oks:
        # If "ok" wasn't saved as literal True/False, just fall back to all rows
        oks = rows

    if trial is not None:
        matches = [r for r in oks if int(r.get("trial", -1)) == int(trial)]
        if not matches:
            raise RuntimeError(f"Trial {trial} not found in CSV.")
        picked = matches[0]
        print(f"[CSV] Selected trial={picked.get('trial')} (forced)")
        return picked

    # Pick best by "score"
    oks_sorted = sorted(oks, key=lambda r: float(r.get("score", float("-inf"))), reverse=True)
    picked = oks_sorted[0]
    print(f"[CSV] Selected BEST by score: trial={picked.get('trial')} score={picked.get('score')}")
    return picked

# --------------------------
# Full-video rendering
# --------------------------
def render_full_video(video_path: str,
                      kn: Dict[str, Any],
                      weights: str,
                      out_path: str,
                      target_classes: List[str]) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 0:
        src_fps = 25.0

    # Read the first frame to initialize sizes/writer
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("No frames in source video.")

    # Build pipeline elements
    yolo = YoloRunner(weights_path=weights)
    print(f"[Debug] weather_on config value: {kn.get('weather_on', 0)} (type: {type(kn.get('weather_on', 0))})")
    weather_enabled = bool(int(kn.get("weather_on", 0)))
    print(f"[Debug] weather_enabled converted to: {weather_enabled}")
    weather = WeatherRunner(enabled=False)
    monitor = build_monitor(kn)

    # Apply limit_resolution (if any) to the *first* frame to determine writer size
    f0 = frame
    if int(kn.get("limit_resolution", 0)) == 1:
        h, w = f0.shape[:2]
        mw, mh = int(kn["max_w"]), int(kn["max_h"])
        if w > mw or h > mh:
            r = min(mw / w, mh / h)
            f0 = cv2.resize(f0, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

    # Prepare first processed pair (to size the writer exactly)
    enhanced0 = weather(f0)
    min_score = None if float(kn.get("init_min_score", -1.0)) < 0 else float(kn["init_min_score"])
    corrected0, meta0 = monitor.preprocess_for_yolo(enhanced0, min_score=min_score)
    det_raw0 = yolo.infer(enhanced0, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))
    det_cor0 = yolo.infer(corrected0, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))

    left0  = YoloRunner.draw(enhanced0, det_raw0)
    right0 = YoloRunner.draw(corrected0, det_cor0)

    hud0 = f"maskNZ={meta0.get('mask_nonzero',0)} | active={meta0.get('active_tiles',0)} | wx={'on' if int(kn.get('weather_on',0)) else 'off'}"
    cv2.putText(right0, hud0, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    if int(kn.get("start_with_grid", 0)) == 1:
        right0 = draw_grid(right0, grid=(int(kn.get("grid_x",8)), int(kn.get("grid_y",6))), color=(80,80,80))

    side_h, side_w = left0.shape[0], left0.shape[1] + right0.shape[1]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (side_w, side_h))
    writer.write(np.hstack([left0, right0]))

    # Process remaining frames
    import time
    start_time = time.time()
    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        f = frame
        if int(kn.get("limit_resolution", 0)) == 1:
            h, w = f.shape[:2]
            mw, mh = int(kn["max_w"]), int(kn["max_h"])
            if w > mw or h > mh:
                r = min(mw / w, mh / h)
                f = cv2.resize(f, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

        enhanced = weather(f)
        corrected, meta = monitor.preprocess_for_yolo(enhanced, min_score=min_score)

        det_raw = yolo.infer(enhanced, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))
        det_cor = yolo.infer(corrected, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))

        left  = YoloRunner.draw(f, det_raw)
        right = YoloRunner.draw(corrected, det_cor)

        hud = f"maskNZ={meta.get('mask_nonzero',0)} | active={meta.get('active_tiles',0)} | wx={'on' if int(kn.get('weather_on',0)) else 'off'}"
        cv2.putText(right, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
        if int(kn.get("start_with_grid", 0)) == 1:
            right = draw_grid(right, grid=(int(kn.get("grid_x",8)), int(kn.get("grid_y",6))), color=(80,80,80))

        # If sizes drift (shouldn't), resize to match first pair
        if left.shape[0] != side_h:
            left = cv2.resize(left, (left.shape[1]*side_h//left.shape[0], side_h))
        if right.shape[0] != side_h:
            right = cv2.resize(right, (right.shape[1]*side_h//right.shape[0], side_h))
        if left.shape[1] + right.shape[1] != side_w:
            # pad/truncate to match writer width
            canvas = np.zeros((side_h, side_w, 3), dtype=np.uint8)
            lw = min(left.shape[1], side_w//2)
            rw = min(right.shape[1], side_w - lw)
            canvas[:, :lw] = left[:, :lw]
            canvas[:, lw:lw+rw] = right[:, :rw]
            writer.write(canvas)
        else:
            writer.write(np.hstack([left, right]))

        frame_idx += 1
        if frame_idx % 250 == 0:
            print(f"[render] wrote {frame_idx} frames...")

    writer.release()
    cap.release()
    print(f"[full] wrote {out_path} ({frame_idx} frames @ {src_fps:.2f} fps)")

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Render full video from optimizer CSV config.")
    ap.add_argument("--video", required=True, help="Path to source video (full clip).")
    ap.add_argument("--csv", required=True, help="Path to optimizer_log.csv.")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path.")
    ap.add_argument("--out", default="results_opt/full_best.mp4", help="Output mp4 path.")
    ap.add_argument("--classes", default="car,truck,bus,motorcycle,bicycle,person", help="Comma list of target classes.")
    ap.add_argument("--trial", type=int, default=None, help="Force a specific trial id from CSV.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # 1) Read CSV, pick config
    kn = load_best_or_trial(args.csv, args.trial)

    # Sanity for keys present in grid mode
    # (max_w, max_h required if limit_resolution==1)
    if int(kn.get("limit_resolution", 0)) == 1:
        for k in ("max_w", "max_h"):
            if k not in kn:
                raise RuntimeError(f"CSV missing required key '{k}' when limit_resolution=1")

    # 2) Render full video
    render_full_video(args.video, kn, args.weights, args.out, target_classes)

    # 3) Save the used config for provenance
    cfg_out = os.path.splitext(args.out)[0] + ".config.json"
    with open(cfg_out, "w") as f:
        json.dump(kn, f, indent=2)
    print(f"[config] saved to {cfg_out}")

if __name__ == "__main__":
    main()