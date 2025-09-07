#!/usr/bin/env python3
# MAIN_CAM_WEATHER_YOLO_OPTIMIZER_fast.py
# Speed-ups:
#  - Single video decode (frames cached in RAM)
#  - Single YOLO load (per-frame conf/iou overrides)
#  - Optional global downscale + sweep stride
#  - ETA + stable FPS reporting

import os, cv2, time, json, csv, math, random, argparse, itertools, types
import numpy as np
from typing import Any, Dict, List, Tuple

from CAMFINALREAL import CameraMonitor  # unchanged

# Weather (optional)
try:
    from WeatherModel import process_frame as weather_process_frame
    HAVE_WEATHER = True
except Exception:
    HAVE_WEATHER = False
    weather_process_frame = None

# Optional shim for externally loaded YOLO
try:
    import yolo_loaded  # exposes yolo_predict(rgb) or yolo_model
except Exception:
    yolo_loaded = None

# --------------------------
# Frame cache
# --------------------------
def load_frames(path: str, max_frames: int = 0, stride: int = 0,
                pre_resize_w: int = 0, pre_resize_h: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames, i, used = [], 0, 0
    ok, f = cap.read()
    while ok:
        take = (stride <= 0) or (i % (stride + 1) == 0)
        if take:
            if pre_resize_w > 0 and pre_resize_h > 0:
                h, w = f.shape[:2]
                if w > pre_resize_w or h > pre_resize_h:
                    r = min(pre_resize_w / w, pre_resize_h / h)
                    f = cv2.resize(f, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
            frames.append(f.copy())
            used += 1
            if max_frames > 0 and used >= max_frames:
                break
        i += 1
        ok, f = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError("No frames loaded from video.")
    return frames

# --------------------------
# YOLO runner (single load, per-call conf/iou)
# --------------------------
class YoloRunner:
    def __init__(self, weights_path="yolov8n.pt"):
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
                # decide device/precision
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
            # pass conf/iou per call; set device/half once
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
# Weather wrapper
# --------------------------
class WeatherRunner:
    def __init__(self, enabled=True):
        self.enabled = HAVE_WEATHER and enabled

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame_bgr
        try:
            enhanced, _ = weather_process_frame(frame_bgr)
            return enhanced if enhanced is not None else frame_bgr
        except Exception:
            self.enabled = False
            return frame_bgr

# --------------------------
# Metrics
# --------------------------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    if inter <= 0: return 0.0
    aa = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    return inter / max(1e-6, aa + bb - inter)

def unmatched_count(new, base, iou_thr=0.5, classes=None) -> int:
    used = [False]*len(base)
    cnt = 0
    for d in new:
        if classes is not None and d.get("name") not in classes and d.get("cls") not in classes:
            continue
        bb = d['xyxy']; matched = False
        for j,db in enumerate(base):
            if used[j]: continue
            if classes is not None and db.get("name") not in classes and db.get("cls") not in classes:
                continue
            if iou(bb, db['xyxy']) >= iou_thr:
                used[j] = True; matched = True; break
        if not matched: cnt += 1
    return cnt

def mean_conf(dets, classes=None) -> float:
    vals = []
    for d in dets:
        if classes is not None and d.get("name") not in classes and d.get("cls") not in classes:
            continue
        vals.append(float(d.get("conf", 0.0)))
    return float(np.mean(vals)) if vals else 0.0

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
# Param spaces (unchanged)
# --------------------------
def param_space_grid():
    return {
        "limit_resolution": [1, 0],
        "max_w": [640, 800, 960],
        "max_h": [480, 540, 720],
        "start_with_grid": [1, 0],
        "yolo_conf": [0.2, 0.25, 0.35],
        "yolo_iou":  [0.45, 0.5],
        "weather_on": [1, 0],
        "grid_x": [8], "grid_y": [6],
        "persistence_hi": [0.65, 0.75, 0.85],
        "ratio_lap": [0.5, 0.55, 0.6],
        "ratio_edge": [0.5, 0.55, 0.6],
        "ratio_contr": [0.6, 0.7, 0.9],
        "global_min_edges": [0.01, 0.05, 0.1],
        "global_min_lap": [10.0, 20.0, 30.0],
        "freeze_margin": [0.8, 0.85, 0.9],
        "seq_len": [20, 30],
        "pixel_mask_decay": [0.95, 0.98, 0.995],
        "healthy_deactivate": [30, 50],
        "ewma_alpha": [0.1, 0.2],
        "decay_pixel": [0.95, 0.98],
        "decay_tile": [0.9, 0.95, 0.98],
        "delta_thresh": [15, 20, 25],
        "hot_val": [240, 245, 250],
        "dead_val": [5, 10, 15],
        "var_thresh": [1.5, 2.0, 3.0],
        "min_tile_area": [300, 400, 600],
        "clahe_clip": [-1.0, 2.0],
        "clahe_tile_w": [-1, 8], "clahe_tile_h": [-1, 8],
        "persist_mask_thr": [0.5, 0.6, 0.7],
        "active_healthy_thr": [0.1, 0.15, 0.2],
        "auto_baseline": [1, 0],
        "baseline_every_s": [60.0, 120.0, 240.0],
        "init_min_score": [-1.0, 0.7, 0.85, 0.95],
    }

def param_sample_random() -> Dict[str, Any]:
    return {
        "limit_resolution": random.choice([1, 0]),
        "max_w": random.choice([640, 800, 960, 1280]),
        "max_h": random.choice([480, 540, 720, 800]),
        "start_with_grid": random.choice([1, 0]),
        "yolo_conf": round(random.uniform(0.18, 0.4), 2),
        "yolo_iou": round(random.uniform(0.45, 0.6), 2),
        "weather_on": random.choice([1, 0]),
        "grid_x": 8, "grid_y": 6,
        "persistence_hi": round(random.uniform(0.6, 0.9), 2),
        "ratio_lap": round(random.uniform(0.45, 0.65), 2),
        "ratio_edge": round(random.uniform(0.45, 0.65), 2),
        "ratio_contr": round(random.uniform(0.6, 0.95), 2),
        "global_min_edges": round(random.uniform(0.005, 0.12), 3),
        "global_min_lap": round(random.uniform(8.0, 40.0), 1),
        "freeze_margin": round(random.uniform(0.75, 0.92), 2),
        "seq_len": random.choice([20, 30, 40]),
        "pixel_mask_decay": round(random.uniform(0.93, 0.997), 3),
        "healthy_deactivate": random.choice([20, 30, 50, 60]),
        "ewma_alpha": round(random.uniform(0.05, 0.25), 2),
        "decay_pixel": round(random.uniform(0.94, 0.99), 2),
        "decay_tile": round(random.uniform(0.88, 0.98), 2),
        "delta_thresh": random.choice([12, 15, 20, 25, 30]),
        "hot_val": random.choice([238, 240, 245, 250]),
        "dead_val": random.choice([5, 8, 10, 12, 15]),
        "var_thresh": round(random.uniform(1.2, 3.5), 2),
        "min_tile_area": random.choice([256, 300, 400, 600]),
        "clahe_clip": random.choice([-1.0, 2.0, 2.5]),
        "clahe_tile_w": random.choice([-1, 8, 12]),
        "clahe_tile_h": random.choice([-1, 8, 12]),
        "persist_mask_thr": round(random.uniform(0.5, 0.75), 2),
        "active_healthy_thr": round(random.uniform(0.08, 0.25), 2),
        "auto_baseline": random.choice([1, 0]),
        "baseline_every_s": random.choice([45.0, 60.0, 120.0, 240.0]),
        "init_min_score": random.choice([-1.0, 0.7, 0.8, 0.9, 0.95]),
    }

# --------------------------
# Numeric overrides (unchanged)
# --------------------------
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
# Trial evaluation (uses cached frames + single YOLO)
# --------------------------
def run_trial(prepared_frames: List[np.ndarray],
              kn: Dict[str, Any],
              yolo: YoloRunner,
              target_classes: List[str]) -> Dict[str, Any]:

    weather = WeatherRunner(enabled=bool(kn["weather_on"]))
    monitor = build_monitor(kn)

    total_score = 0.0
    total_new = 0.0
    total_conf_gain = 0.0
    total_mask_nz = 0.0
    total_active = 0.0
    frames_used = 0

    # baseline cadence emulation (optional)
    last_baseline_refresh = time.time()
    auto_baseline = bool(kn["auto_baseline"])
    baseline_every_s = float(kn["baseline_every_s"])

    t0 = time.time()
    for frame in prepared_frames:
        # Resolution limit per trial
        if int(kn["limit_resolution"]) == 1:
            h, w = frame.shape[:2]
            mw, mh = int(kn["max_w"]), int(kn["max_h"])
            if w > mw or h > mh:
                r = min(mw / w, mh / h)
                frame = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

        enhanced = weather(frame)
        min_score = None if float(kn["init_min_score"]) < 0 else float(kn["init_min_score"])
        corrected, meta = monitor.preprocess_for_yolo(enhanced, min_score=min_score)

        if auto_baseline and (time.time() - last_baseline_refresh) >= baseline_every_s:
            try:
                new_base = monitor.updater.update_baseline_from_logs(monitor.logger)
                if new_base is not None:
                    monitor.drift_detector.set_baseline(new_base)
                last_baseline_refresh = time.time()
            except Exception:
                pass

        det_raw = yolo.infer(enhanced, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))
        det_cor = yolo.infer(corrected, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))

        new_dets = unmatched_count(det_cor, det_raw, iou_thr=0.5, classes=target_classes)
        conf_gain = max(0.0, mean_conf(det_cor, target_classes) - mean_conf(det_raw, target_classes))
        mask_nz = int(meta.get("mask_nonzero", 0))
        active_tiles = int(meta.get("active_tiles", 0))

        frame_score = new_dets + 0.5 * conf_gain - 0.001 * mask_nz - 0.05 * active_tiles
        total_score += frame_score
        total_new += new_dets
        total_conf_gain += conf_gain
        total_mask_nz += mask_nz
        total_active += active_tiles
        frames_used += 1

    dt = max(1e-6, time.time() - t0)
    fps = frames_used / dt
    if frames_used == 0:
        return {"ok": False, "error": "No frames evaluated."}

    return {
        "ok": True,
        "frames": frames_used,
        "score": total_score / frames_used,
        "avg_new": total_new / frames_used,
        "avg_conf_gain": total_conf_gain / frames_used,
        "avg_mask_nz": total_mask_nz / frames_used,
        "avg_active": total_active / frames_used,
        "fps": fps
    }

# --------------------------
# Render best with cached frames + single YOLO
# --------------------------
def render_best(prepared_frames: List[np.ndarray],
                kn: Dict[str, Any],
                yolo: YoloRunner,
                out_path: str,
                target_classes: List[str]):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    first = prepared_frames[0]
    side_h, side_w = first.shape[0], first.shape[1] * 2
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (side_w, side_h))

    weather = WeatherRunner(enabled=bool(kn["weather_on"]))
    monitor = build_monitor(kn)

    for frame in prepared_frames:
        f = frame
        if int(kn["limit_resolution"]) == 1:
            h, w = f.shape[:2]
            mw, mh = int(kn["max_w"]), int(kn["max_h"])
            if w > mw or h > mh:
                r = min(mw / w, mh / h)
                f = cv2.resize(f, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

        enhanced = weather(f)
        min_score = None if float(kn["init_min_score"]) < 0 else float(kn["init_min_score"])
        corrected, meta = monitor.preprocess_for_yolo(enhanced, min_score=min_score)
        det_raw = yolo.infer(enhanced, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))
        det_cor = yolo.infer(corrected, conf=float(kn["yolo_conf"]), iou=float(kn["yolo_iou"]))

        left  = YoloRunner.draw(enhanced, det_raw)
        right = YoloRunner.draw(corrected, det_cor)

        hud = f"maskNZ={meta.get('mask_nonzero',0)} | active={meta.get('active_tiles',0)} | wx={'on' if kn['weather_on'] else 'off'}"
        cv2.putText(right, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
        if int(kn["start_with_grid"]) == 1:
            right = draw_grid(right, grid=(int(kn["grid_x"]), int(kn["grid_y"])), color=(80,80,80))

        writer.write(np.hstack([left, right]))

    writer.release()
    print(f"[best] wrote {out_path} ({len(prepared_frames)} frames)")

# --------------------------
# Trial generation (unchanged)
# --------------------------
def generate_trials(mode: str, max_trials: int) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    if mode == "grid":
        space = param_space_grid()
        groups = [
            ["limit_resolution","max_w","max_h","start_with_grid","weather_on","yolo_conf","yolo_iou","init_min_score"],
            ["persistence_hi","ratio_lap","ratio_edge","ratio_contr","global_min_edges","global_min_lap","freeze_margin"],
            ["seq_len","pixel_mask_decay","healthy_deactivate","ewma_alpha","decay_pixel","decay_tile"],
            ["delta_thresh","hot_val","dead_val","var_thresh","min_tile_area",
             "clahe_clip","clahe_tile_w","clahe_tile_h","persist_mask_thr","active_healthy_thr",
             "auto_baseline","baseline_every_s"],
        ]
        combos = [{}]
        for g in groups:
            items = [(k, space[k]) for k in g]
            block = []
            for values in itertools.product(*[v for _,v in items]):
                d = {}
                for (k,_), val in zip(items, values): d[k]=val
                block.append(d)
            random.shuffle(block)
            block = block[:min(120, len(block))]
            new = []
            for base in combos:
                for b in block:
                    dd = dict(base); dd.update(b); new.append(dd)
            combos = new
            random.shuffle(combos)
            combos = combos[:min(max_trials, len(combos))]
        trials = combos[:max_trials]
    else:
        for _ in range(max_trials):
            trials.append(param_sample_random())
    for t in trials:
        t.setdefault("grid_x", 8); t.setdefault("grid_y", 6)
    return trials

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Optimize Weather→CAM→YOLO over a test video (fast).")
    ap.add_argument("--video", required=True, help="Path to short road video (e.g., 10–20s).")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path.")
    ap.add_argument("--mode", choices=["grid","random"], default="grid", help="Search mode.")
    ap.add_argument("--trials", type=int, default=200, help="Total trials to run.")
    # Fast sweep knobs (applied ONCE to the cached frames)
    ap.add_argument("--frames", type=int, default=0, help="Max frames to load for sweep/render (0 = all in clip).")
    ap.add_argument("--skip", type=int, default=2, help="Use 1 of every (skip+1) frames during the sweep cache.")
    ap.add_argument("--pre_w", type=int, default=960, help="Global pre-resize width for cache (0 = none).")
    ap.add_argument("--pre_h", type=int, default=540, help="Global pre-resize height for cache (0 = none).")
    ap.add_argument("--classes", default="car,truck,bus,motorcycle,bicycle,person", help="Comma list of target classes.")
    ap.add_argument("--outdir", default="results_opt", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # 1) Load frames ONCE (with sweep stride + global pre-resize)
    frames = load_frames(
        args.video,
        max_frames=max(0, args.frames),
        stride=max(0, args.skip),
        pre_resize_w=max(0, args.pre_w),
        pre_resize_h=max(0, args.pre_h)
    )
    print(f"[cache] Loaded {len(frames)} frames (skip={args.skip}, pre={args.pre_w}x{args.pre_h}).")

    # 2) Load YOLO ONCE
    yolo = YoloRunner(weights_path=args.weights)

    # 3) Prepare trials
    trials = generate_trials(args.mode, max(1, args.trials))
    print(f"Prepared {len(trials)} trial(s). Starting sweep on {args.video}...")

    logs: List[Dict[str, Any]] = []
    best = None
    start = time.time()

    for i,kn in enumerate(trials, 1):
        t_trial0 = time.time()
        m = run_trial(
            prepared_frames=frames,
            kn=kn,
            yolo=yolo,
            target_classes=target_classes
        )
        row = {"trial": i, **kn, **m}
        logs.append(row)

        # ETA
        elapsed = time.time() - start
        eta = elapsed / i * (len(trials) - i)

        if m.get("ok"):
            if best is None or m["score"] > best["score"]:
                best = {"trial": i, **kn, **m}
            print(f"[{i}/{len(trials)}] "
                  f"score={m['score']:.3f} new={m['avg_new']:.3f} conf+={m['avg_conf_gain']:.3f} "
                  f"maskNZ={m['avg_mask_nz']:.1f} active={m['avg_active']:.2f} fps={m['fps']:.1f} "
                  f"| trial={time.time()-t_trial0:.1f}s ETA={eta/60:.1f}m")
        else:
            print(f"[{i}/{len(trials)}] ERROR: {m.get('error')} | ETA={eta/60:.1f}m")

    # Save logs
    csv_path = os.path.join(args.outdir, "optimizer_log.csv")
    json_path = os.path.join(args.outdir, "optimizer_log.json")
    if logs:
        keys = sorted(set().union(*[row.keys() for row in logs]))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in logs: w.writerow(r)
        with open(json_path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Saved logs: {csv_path} | {json_path}")

    # Print best & render
    oks = [r for r in logs if r.get("ok")]
    oks.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== TOP 5 CONFIGS ===")
    for j,row in enumerate(oks[:5], 1):
        print(f"#{j} trial={row['trial']} score={row['score']:.3f} new={row['avg_new']:.3f} "
              f"conf+={row['avg_conf_gain']:.3f} maskNZ={row['avg_mask_nz']:.1f} "
              f"active={row['avg_active']:.2f} fps={row['fps']:.1f}")

    if best and best.get("ok"):
        best_path = os.path.join(args.outdir, "best_run.mp4")
        print("\nRendering best config to video...")
        render_best(frames, best, yolo, best_path, target_classes=target_classes)
        best_cfg_path = os.path.join(args.outdir, "best_config.json")
        with open(best_cfg_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"BEST CONFIG saved to {best_cfg_path}")
    else:
        print("No successful trials to render.")

if __name__ == "__main__":
    main()
