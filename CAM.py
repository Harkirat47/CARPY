# monitor.py
# Clean, modular camera blind-spot + drift monitor with optional depth (MiDaS-ready).
# Python 3.10+ recommended.

from __future__ import annotations
import cv2
import numpy as np
import time, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque
from datetime import datetime

# ======= Utilities =======

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def shannon_entropy(gray: np.ndarray) -> float:
    # gray uint8 expected
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (gray.size + 1e-9)
    p = p[p>0]
    return float(-np.sum(p * np.log2(p)))

def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / (gray.size + 1e-9)

def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def exposure_stats(gray: np.ndarray) -> Tuple[float, float]:
    # mean brightness, contrast (std)
    return float(np.mean(gray)), float(np.std(gray))

def depth_variance(depth: np.ndarray) -> float:
    # expects float32 depth in meters or arbitrary scale; NaNs handled
    d = depth[np.isfinite(depth)]
    return float(np.var(d)) if d.size else 0.0

def depth_edge_density(depth: np.ndarray) -> float:
    if depth is None:
        return 0.0
    d = depth.copy()
    d[~np.isfinite(d)] = 0.0
    gradx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gradx, grady)
    thr = float(np.nanmedian(mag) + 2*np.nanstd(mag))
    edges = (mag > thr).astype(np.uint8)
    return float(np.count_nonzero(edges)) / (edges.size + 1e-9)

# ======= Configs =======

@dataclass
class GridConfig:
    cols: int = 8
    rows: int = 6
    min_cell_px: int = 4000       # ignore very small cells

@dataclass
class BlindspotConfig:
    # These are *soft* priors; dynamic (frame-level) quantiles refine them each frame.
    blur_low: float = 80.0        # Laplacian variance considered "low"
    entropy_low: float = 5.0      # "textureless"
    edge_low: float = 0.02
    bright_low: float = 40.0
    bright_high: float = 200.0
    contrast_low: float = 20.0
    depth_var_low: float = 0.001  # small variation means flat depth (optional)
    depth_edge_high: float = 0.08

@dataclass
class DriftConfig:
    warmup_frames: int = 30
    ewma_alpha: float = 0.05      # adaptive baseline smoothing
    mad_scale: float = 4.0        # threshold = median + mad_scale * MAD
    feature_bins: int = 32        # color histogram bins per channel

@dataclass
class IOConfig:
    log_dir: Path = Path("drift_logs")
    calib_dir: Path = Path("calibration_photos")
    cooldown_s: float = 5.0

# ======= Frame Source =======

class FrameSource:
    def __init__(self, cam_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(cam_index)
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

# ======= Depth Provider (MiDaS-ready) =======

class DepthProvider:
    """Interface. Implement estimate(frame_bgr) -> depth_float32 with MiDaS later."""
    def estimate(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        return None  # no depth by default

# ======= Feature Extractor for Drift (lightweight & safe) =======

class HistogramFeatures:
    """RGB histograms concatenated (L1-normalized). Cheap and robust for ‘scene drift’."""
    def __init__(self, bins: int = 32):
        self.bins = bins

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        chans = cv2.split(frame_bgr)
        feats = []
        for ch in chans:
            h = cv2.calcHist([ch],[0],None,[self.bins],[0,256]).astype(np.float32).ravel()
            h /= (np.sum(h) + 1e-9)
            feats.append(h)
        return np.concatenate(feats, axis=0)  # shape = 3*bins

# ======= Drift Monitor (EWMA + MAD thresholding) =======

class DriftMonitor:
    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.fe = HistogramFeatures(cfg.feature_bins)
        self.baseline: Optional[np.ndarray] = None
        self.residuals = deque(maxlen=300)
        self.num_seen = 0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # L2 in histogram space (could switch to Bhattacharyya)
        return float(np.linalg.norm(a - b))

    def _mad_threshold(self) -> float:
        if len(self.residuals) < 10:
            return float('inf')  # avoid early triggers
        med = np.median(self.residuals)
        mad = np.median(np.abs(self.residuals - med)) + 1e-9
        return float(med + self.cfg.mad_scale * mad)

    def update_and_check(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        feat = self.fe(frame_bgr)  # (3*bins,)
        if self.baseline is None:
            self.baseline = feat.copy()
            self.num_seen = 1
            return False, 0.0

        # EWMA baseline
        self.baseline = (1 - self.cfg.ewma_alpha) * self.baseline + self.cfg.ewma_alpha * feat
        dist = self.distance(feat, self.baseline)

        # Update residuals (post-warmup)
        self.num_seen += 1
        if self.num_seen > self.cfg.warmup_frames:
            self.residuals.append(dist)
            is_drift = dist > self._mad_threshold()
        else:
            is_drift = False

        return is_drift, dist

# ======= Blindspot Detector =======

@dataclass
class CellMetrics:
    bbox: Tuple[int,int,int,int]
    lapvar: float
    entropy: float
    edgedens: float
    mean: float
    contrast: float
    depth_var: Optional[float] = None
    depth_edge: Optional[float] = None
    reason: Optional[str] = None

class BlindspotDetector:
    def __init__(self, grid_cfg: GridConfig, bs_cfg: BlindspotConfig):
        self.gcfg = grid_cfg
        self.cfg = bs_cfg

    def analyze(self, frame_bgr: np.ndarray, depth: Optional[np.ndarray]) -> List[CellMetrics]:
        h, w = frame_bgr.shape[:2]
        step_x = max(1, w // self.gcfg.cols)
        step_y = max(1, h // self.gcfg.rows)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        cells: List[CellMetrics] = []
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                x2 = min(x + step_x, w)
                y2 = min(y + step_y, h)
                if (x2-x)*(y2-y) < self.gcfg.min_cell_px: 
                    continue

                roi_g = gray[y:y2, x:x2]
                lap = laplacian_var(roi_g)
                ent = shannon_entropy(roi_g)
                edg = edge_density(roi_g)
                mu, sig = exposure_stats(roi_g)

                dvar = dedge = None
                if depth is not None:
                    roi_d = depth[y:y2, x:x2]
                    dvar = depth_variance(roi_d)
                    dedge = depth_edge_density(roi_d)

                cells.append(CellMetrics((x,y,x2,y2), lap, ent, edg, mu, sig, dvar, dedge))

        # Dynamic thresholds via frame quantiles (robust to lighting changes)
        if cells:
            lap_q20 = np.quantile([c.lapvar for c in cells], 0.20)
            ent_q20 = np.quantile([c.entropy for c in cells], 0.20)
            edg_q20 = np.quantile([c.edgedens for c in cells], 0.20)

        blindspots: List[CellMetrics] = []
        for c in cells:
            reasons = []

            # Exposure issues
            if c.mean < min(self.cfg.bright_low, 0.8*lap_q20): reasons.append("under-exposed")
            if c.mean > self.cfg.bright_high: reasons.append("over-exposed")

            # Blur / low detail
            if c.lapvar < min(self.cfg.blur_low, 1.2*lap_q20): reasons.append("defocus/motion blur")
            if c.entropy < min(self.cfg.entropy_low, 1.1*ent_q20): reasons.append("low texture")
            if c.edgedens < min(self.cfg.edge_low, 1.1*edg_q20): reasons.append("few edges")
            if c.contrast < self.cfg.contrast_low: reasons.append("low contrast")

            # Depth-based hints (optional)
            if c.depth_var is not None:
                if c.depth_var < self.cfg.depth_var_low and "low texture" in reasons:
                    reasons.append("flat depth")
                if c.depth_edge is not None and c.depth_edge > self.cfg.depth_edge_high:
                    # high depth edges but low image edges → likely occlusion / glare
                    if c.edgedens < self.cfg.edge_low:
                        reasons.append("possible occlusion/glare")

            if reasons:
                c.reason = ", ".join(sorted(set(reasons)))
                blindspots.append(c)

        return blindspots

# ======= Annotation & Logging =======

class Annotator:
    def draw(self, frame: np.ndarray, blindspots: List[CellMetrics], title: str, drift_dist: float) -> np.ndarray:
        vis = frame.copy()
        for c in blindspots:
            x,y,x2,y2 = c.bbox
            cv2.rectangle(vis, (x,y), (x2,y2), (0,0,255), 2)
            label = c.reason if c.reason else "issue"
            cv2.putText(vis, label, (x+3, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{title}  |  DriftDist={drift_dist:.2f}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        return vis

class EventLogger:
    def __init__(self, io: IOConfig):
        self.io = io
        ensure_dir(io.log_dir)
        self.last_log = 0.0

    def maybe_log(self, frame: np.ndarray, drift_dist: float, blindspots: List[CellMetrics]) -> bool:
        t = time.time()
        if t - self.last_log < self.io.cooldown_s:
            return False
        stamp = now_str()
        base = f"drift_{stamp}_d{drift_dist:.2f}"
        imgp = self.io.log_dir / f"{base}.jpg"
        metap = self.io.log_dir / f"{base}.json"

        cv2.imwrite(str(imgp), frame)
        meta = {
            "timestamp": stamp,
            "drift_distance": drift_dist,
            "blindspots": [
                {"bbox": c.bbox, "reason": c.reason,
                 "lapvar": c.lapvar, "entropy": c.entropy, "edge_density": c.edgedens,
                 "brightness": c.mean, "contrast": c.contrast,
                 "depth_var": c.depth_var, "depth_edge": c.depth_edge}
                for c in blindspots
            ]
        }
        with open(metap, "w") as f:
            json.dump(meta, f, indent=2)
        self.last_log = t
        print(f"⚠️  Logged: {imgp.name}  ({len(blindspots)} blindspots)")
        return True

# ======= Orchestrator =======

class CameraMonitor:
    def __init__(
        self,
        grid_cfg: GridConfig = GridConfig(),
        bs_cfg: BlindspotConfig = BlindspotConfig(),
        drift_cfg: DriftConfig = DriftConfig(),
        io_cfg: IOConfig = IOConfig(),
        depth_provider: Optional[DepthProvider] = None
    ):
        self.src = FrameSource()
        self.depth = depth_provider or DepthProvider()
        self.bs = BlindspotDetector(grid_cfg, bs_cfg)
        self.drift = DriftMonitor(drift_cfg)
        self.ann = Annotator()
        self.logger = EventLogger(io_cfg)
        self.io = io_cfg
        ensure_dir(io_cfg.calib_dir)

    def loop(self):
        title = "Camera Blind-spot Monitor"
        while True:
            frame = self.src.read()
            if frame is None: break

            # Optional depth (MiDaS later)
            depth = self.depth.estimate(frame)  # None by default

            # Drift
            is_drift, dist = self.drift.update_and_check(frame)

            # Blindspots (per-cell)
            blindspots = self.bs.analyze(frame, depth)

            # Draw + show
            vis = self.ann.draw(frame, blindspots, title, dist)
            cv2.imshow("monitor", vis)

            # Log on meaningful events
            if is_drift and len(blindspots) >= 4:
                self.logger.maybe_log(vis, dist, blindspots)

            # Hotkeys
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):  # ESC or q
                break
            if k in (ord('p'), ord('P')):
                path = self.io.calib_dir / f"calib_{now_str()}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"📷 Saved calibration photo → {path}")

        self.src.release()
        cv2.destroyAllWindows()

# ======= CLI =======

if __name__ == "__main__":
    monitor = CameraMonitor()
    monitor.loop()
