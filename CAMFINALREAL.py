import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
from pathlib import Path
import numpy as np
import time
import json
from collections import deque
from datetime import datetime

# ============================================================
# Utilities
# ============================================================

def _edge_density(img):
    e = cv2.Canny(img, 100, 200)
    return float(np.sum(e > 0) / (img.size + 1e-5))

# ============================================================
# Spot Issue Classifier (tiny LSTM + fallback rules)
# ============================================================

class SpotIssueClassifier(nn.Module):
    """
    Tiny LSTM classifier that learns per-tile temporal patterns to classify
    a tile issue as CAMERA vs SETTING. It is trained online when weak labels
    are available; otherwise we fall back to robust rules.

    Per-step features: [r_lap, r_edge, r_contr, global_med_edge, global_med_lap]
    Labels: 0=SETTING, 1=CAMERA
    """
    def __init__(self, input_size=5, hidden=16, num_layers=1, max_mem=200):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.mem_X, self.mem_y = [], []
        self.max_mem = max_mem
        self.num_labels = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    @torch.no_grad()
    def forward(self, seq_feats):
        # seq_feats: (B, T, F)
        out, _ = self.lstm(seq_feats)
        logits = self.head(out[:, -1, :])
        return logits

    def add_labeled_sequence(self, seq_feats_np, label_int):
        self.mem_X.append(torch.tensor(seq_feats_np, dtype=torch.float32))
        self.mem_y.append(int(label_int))
        if len(self.mem_X) > self.max_mem:
            self.mem_X.pop(0); self.mem_y.pop(0)
        self.num_labels += 1

    def train_step(self, batch_size=8, steps=1):
        if len(self.mem_X) < 8:
            return
        self.train()
        for _ in range(steps):
            idx = np.random.choice(len(self.mem_X), size=min(batch_size, len(self.mem_X)), replace=False)
            X = [self.mem_X[i] for i in idx]
            y = [self.mem_y[i] for i in idx]
            maxT = max(x.shape[0] for x in X)
            Xp = []
            for x in X:
                if x.shape[0] < maxT:
                    pad = torch.zeros((maxT - x.shape[0], x.shape[1]), dtype=torch.float32)
                    Xp.append(torch.cat([x, pad], dim=0))
                else:
                    Xp.append(x)
            Xb = torch.stack(Xp, dim=0).to(self.device)  # (B,T,F)
            yb = torch.tensor(y, dtype=torch.long, device=self.device)
            logits = self.forward(Xb)
            loss = self.loss_fn(logits, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.eval()

    @torch.no_grad()
    def predict(self, seq_feats_np):
        # If too few labels, signal to fallback to rules.
        if self.num_labels < 12:
            return (None, 0.0)
        x = torch.tensor(seq_feats_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return (pred, conf)

# ============================================================
# Blindspot Tracker (sensor-anchored, scene-invariant)
# ============================================================

class BlindspotTracker:
    """
    • Per-pixel EWMA mean/variance to detect dead/hot/stuck pixels (sensor-anchored).
    • Per-tile relative metrics (ratios vs frame medians) for smudge detection (scene-invariant).
    • Informative-frame gating: update tile persistence only when global medians are above small thresholds.
    • Freeze-after-fix + hysteresis: after acknowledge, require ~15% worse than freeze level to re-alert.
    • Per-tile sequence buffers → tiny LSTM classifier; robust rules fallback when untrained.
    """

    def __init__(
        self,
        grid=(8, 6),
        ewma_alpha=0.05,
        decay_pixel=0.98,         # pixel score smoothing
        decay_tile=0.95,          # tile score smoothing
        delta_thresh=20,
        hot_val=245, dead_val=10,
        var_thresh=2.0,
        persistence_hi=0.85,
        min_tile_area=400,
        # relative thresholds (scene-invariant)
        ratio_lap_thresh=0.55, ratio_edge_thresh=0.55, ratio_contrast_thresh=0.65,
        # frame informative gates
        global_min_edges=0.01, global_min_lap=30.0,
        # hysteresis
        freeze_margin=0.85,
        # LSTM
        seq_len=30
    ):
        self.grid = grid
        self.ewma_alpha = ewma_alpha
        self.decay_pixel = decay_pixel
        self.decay_tile = decay_tile

        self.delta_thresh = delta_thresh
        self.hot_val = hot_val
        self.dead_val = dead_val
        self.var_thresh = var_thresh

        self.persistence_hi = persistence_hi
        self.min_tile_area = min_tile_area

        self.ratio_lap_thresh = ratio_lap_thresh
        self.ratio_edge_thresh = ratio_edge_thresh
        self.ratio_contrast_thresh = ratio_contrast_thresh

        self.global_min_edges = global_min_edges
        self.global_min_lap = global_min_lap

        self.freeze_margin = freeze_margin

        # state
        self.shape = None
        self.mu = None
        self.var = None
        self.score_hot = None
        self.score_dead = None
        self.score_stuck = None
        gx, gy = self.grid
        self.tile_scores = np.zeros((gy, gx), dtype=np.float32)

        self.last_boxes = []
        self.last_mask = None

        # freeze/hysteresis
        self.tile_fix_level = np.zeros((gy, gx), dtype=np.float32)
        self.tile_frozen = np.zeros((gy, gx), dtype=np.bool_)

        # per-tile sequence buffers
        self.seq_len = seq_len
        self.tile_seq = [[deque(maxlen=seq_len) for _ in range(gx)] for __ in range(gy)]

        # classifier
        self.classifier = SpotIssueClassifier()

        self._last_global = {'med_lap': 0.0, 'med_edge': 0.0, 'med_contr': 0.0}

    # ---------- init helpers ----------

    def _ensure_state(self, gray):
        if self.shape == gray.shape:
            return
        h, w = gray.shape
        self.shape = (h, w)
        self.mu = gray.astype(np.float32).copy()
        self.var = np.full_like(self.mu, 5.0, dtype=np.float32)
        self.score_hot   = np.zeros_like(self.mu, dtype=np.float32)
        self.score_dead  = np.zeros_like(self.mu, dtype=np.float32)
        self.score_stuck = np.zeros_like(self.mu, dtype=np.float32)
        gx, gy = self.grid
        self.tile_scores = np.zeros((gy, gx), dtype=np.float32)

    # ---------- pixel defects (EWMA + neighbor) ----------

    def _pixel_defects(self, gray):
        med = cv2.medianBlur(gray, 3)
        diff = cv2.absdiff(gray, med)

        alpha = self.ewma_alpha
        mu_prev = self.mu
        mu_new = (1 - alpha) * self.mu + alpha * gray
        self.var = (1 - alpha) * (self.var + (gray - mu_prev) * (gray - mu_new)) + 1e-6
        self.mu = mu_new

        hot_now   = (gray >= self.hot_val)  & (diff >= self.delta_thresh)
        dead_now  = (gray <= self.dead_val) & (diff >= self.delta_thresh)
        stuck_now = (self.var <= self.var_thresh) & (diff >= self.delta_thresh)

        d = self.decay_pixel
        self.score_hot   = (self.score_hot  * d) + (hot_now.astype(np.float32))   * (1 - d)
        self.score_dead  = (self.score_dead * d) + (dead_now.astype(np.float32))  * (1 - d)
        self.score_stuck = (self.score_stuck* d) + (stuck_now.astype(np.float32)) * (1 - d)

        mask = ((self.score_hot > self.persistence_hi) |
                (self.score_dead > self.persistence_hi) |
                (self.score_stuck > self.persistence_hi)).astype(np.uint8) * 255
        return mask

    # ---------- scene-invariant tile metrics ----------

    def _global_tile_stats(self, gray):
        h, w = gray.shape
        gx, gy = self.grid
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)

        laps, edges, contrs = [], [], []
        for j in range(gy):
            for i in range(gx):
                x0 = i * step_x; y0 = j * step_y
                x1 = w if i == gx - 1 else (i + 1) * step_x
                y1 = h if j == gy - 1 else (j + 1) * step_y
                region = gray[y0:y1, x0:x1]
                lap = cv2.Laplacian(region, cv2.CV_64F).var()
                ed  = _edge_density(region)
                ct  = float(np.std(region))
                laps.append(lap); edges.append(ed); contrs.append(ct)

        med_lap   = float(np.median(laps))
        med_edges = float(np.median(edges))
        med_contr = float(np.median(contrs))
        self._last_global = {'med_lap': med_lap, 'med_edge': med_edges, 'med_contr': med_contr}
        return med_lap, med_edges, med_contr

    def _tile_metrics(self, gray):
        h, w = gray.shape
        gx, gy = self.grid
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)

        med_lap, med_edges, med_contr = self._global_tile_stats(gray)
        frame_informative = (med_edges >= self.global_min_edges) or (med_lap >= self.global_min_lap)

        boxes = []
        d = self.decay_tile

        for j in range(gy):
            for i in range(gx):
                x0 = i * step_x; y0 = j * step_y
                x1 = w if i == gx - 1 else (i + 1) * step_x
                y1 = h if j == gy - 1 else (j + 1) * step_y
                if (x1 - x0) * (y1 - y0) < self.min_tile_area:
                    continue

                region = gray[y0:y1, x0:x1]
                lap = cv2.Laplacian(region, cv2.CV_64F).var()
                ed  = _edge_density(region)
                ct  = float(np.std(region))

                r_lap   = lap / (med_lap + 1e-6)
                r_edge  = ed  / (med_edges + 1e-6)
                r_contr = ct  / (med_contr + 1e-6)

                smudge_now = (
                    (r_lap   < self.ratio_lap_thresh) and
                    (r_edge  < self.ratio_edge_thresh) and
                    (r_contr < self.ratio_contrast_thresh)
                )

                # Update persistence only when frame is informative
                if frame_informative:
                    self.tile_scores[j, i] = (self.tile_scores[j, i] * d) + (1 - d) * (1.0 if smudge_now else 0.0)

                # per-tile sequence for LSTM (stable, relative features + global context)
                self.tile_seq[j][i].append([r_lap, r_edge, r_contr, med_edges, med_lap])

                # Freeze-after-fix hysteresis: only re-alert if ~15% worse than frozen level
                cur_level = min(r_lap, r_edge, r_contr)  # smaller is worse
                can_flag = True
                if self.tile_frozen[j, i]:
                    prev = self.tile_fix_level[j, i] if self.tile_fix_level[j, i] > 0 else 1.0
                    can_flag = (cur_level < prev * self.freeze_margin)

                if self.tile_scores[j, i] > self.persistence_hi and can_flag:
                    # try LSTM; else fallback to rules
                    seq = np.array(self.tile_seq[j][i], dtype=np.float32)
                    pred, conf = self.classifier.predict(seq)
                    if pred is None:
                        # Heuristic: if frame informative and the tile's current combined level is quite low → camera.
                        pred_label = "camera" if (frame_informative and cur_level < 0.5) else "setting"
                        pred_conf = 0.55
                    else:
                        pred_label = "camera" if pred == 1 else "setting"
                        pred_conf = float(conf)

                    boxes.append({
                        "bbox": (x0, y0, x1, y1),
                        "kind": "smudge",
                        "score": float(self.tile_scores[j, i]),
                        "why": {
                            "r_lap": round(float(r_lap), 3),
                            "r_edge": round(float(r_edge), 3),
                            "r_contr": round(float(r_contr), 3),
                            "med_lap": round(med_lap, 2),
                            "med_edge": round(med_edges, 4),
                            "med_contr": round(med_contr, 2),
                            "issue": pred_label,
                            "issue_conf": round(pred_conf, 3)
                        }
                    })

        return boxes

    # ---------- public ----------

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._ensure_state(gray)

        pixel_mask = self._pixel_defects(gray)
        boxes = self._tile_metrics(gray)

        # aggregate pixel blobs to boxes
        if np.any(pixel_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean = cv2.morphologyEx(pixel_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            num, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
            for k in range(1, num):
                x, y, w, h, area = stats[k]
                if area < 9:
                    continue
                sub_hot   = self.score_hot[y:y+h, x:x+w].mean()
                sub_dead  = self.score_dead[y:y+h, x:x+w].mean()
                sub_stuck = self.score_stuck[y:y+h, x:x+w].mean()
                kind = max([("hot_pixel", sub_hot), ("dead_pixel", sub_dead), ("stuck_pixel", sub_stuck)],
                           key=lambda t: t[1])[0]
                score = max(sub_hot, sub_dead, sub_stuck)
                boxes.append({
                    "bbox": (int(x), int(y), int(x + w), int(y + h)),
                    "kind": kind,
                    "score": round(float(score), 3),
                    "why": {
                        "avg_hot": round(float(sub_hot), 3),
                        "avg_dead": round(float(sub_dead), 3),
                        "avg_stuck": round(float(sub_stuck), 3),
                        "issue": "camera", "issue_conf": 0.9
                    }
                })

        self.last_boxes = boxes
        self.last_mask = pixel_mask
        return boxes, pixel_mask

    def get_persistent_mask(self, min_score=None):
        if min_score is None:
            min_score = self.persistence_hi
        if self.score_hot is None:
            return np.zeros((1, 1), dtype=np.uint8)
        mask = ((self.score_hot  > min_score) |
                (self.score_dead > min_score) |
                (self.score_stuck> min_score)).astype(np.uint8) * 255
        return mask

    def get_persistent_boxes(self, min_score=None):
        if min_score is None:
            min_score = self.persistence_hi
        return [b for b in self.last_boxes if b.get("score", 0.0) >= min_score]

    def camera_health(self):
        if self.last_mask is None:
            return {"area_masked_pct": 0.0, "tiles_flagged": 0, "needs_reoptimize": False}
        area_masked = float(np.count_nonzero(self.last_mask))
        total = float(self.last_mask.size)
        pct = 100.0 * area_masked / (total + 1e-9)
        tiles_flagged = int(np.sum(self.tile_scores > self.persistence_hi))
        needs = (pct > 2.0) or (tiles_flagged >= 3)
        return {"area_masked_pct": round(pct, 2), "tiles_flagged": tiles_flagged, "needs_reoptimize": bool(needs)}

    # ---------- acknowledge / labeling / unfreeze ----------

    def acknowledge_tiles(self, boxes):
        """Freeze acknowledged smudge tiles + label LSTM as CAMERA."""
        gx, gy = self.grid
        if self.shape is None:
            return
        h, w = self.shape
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)

        for b in boxes:
            if b.get("kind") != "smudge":
                continue
            (x0, y0, _, _) = b["bbox"]
            i = min(int(x0 // step_x), gx - 1)
            j = min(int(y0 // step_y), gy - 1)
            why = b.get("why", {})
            cur_level = min(why.get("r_lap", 1.0), why.get("r_edge", 1.0), why.get("r_contr", 1.0))
            if not self.tile_frozen[j, i] or (cur_level > self.tile_fix_level[j, i]):
                self.tile_fix_level[j, i] = cur_level
            self.tile_frozen[j, i] = True

            seq = np.array(self.tile_seq[j][i], dtype=np.float32)
            if seq.shape[0] >= 8:
                self.classifier.add_labeled_sequence(seq, label_int=1)  # camera

        self.classifier.train_step(batch_size=8, steps=2)

    def label_setting_issue(self, boxes):
        """Add LSTM labels = SETTING for selected tiles (scene/lighting)."""
        gx, gy = self.grid
        if self.shape is None:
            return
        h, w = self.shape
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)

        for b in boxes:
            if b.get("kind") != "smudge":
                continue
            (x0, y0, _, _) = b["bbox"]
            i = min(int(x0 // step_x), gx - 1)
            j = min(int(y0 // step_y), gy - 1)
            seq = np.array(self.tile_seq[j][i], dtype=np.float32)
            if seq.shape[0] >= 8:
                self.classifier.add_labeled_sequence(seq, label_int=0)  # setting

        self.classifier.train_step(batch_size=8, steps=2)

    def clear_freeze_if_recovered(self, strong=True):
        """Unfreeze tiles that have been healthy for a while."""
        if self.tile_scores is None:
            return
        thresh = 0.15 if strong else 0.3
        healthy = (self.tile_scores < thresh)
        self.tile_frozen[healthy] = False

# ============================================================
# Blindspot Corrector (persistent corrections + pixel mask decay)
# ============================================================

class BlindspotCorrector:
    """
    Maintains:
      • active_tiles: tiles that get corrected every frame
      • tile_healthy_count: consecutive healthy frames counter (auto-deactivate)
      • persistent_pixel_mask: decays with factor each frame; always inpainted
    """
    def __init__(self, tracker: BlindspotTracker,
                 healthy_frames_to_deactivate=30,
                 pixel_mask_decay=0.98):
        self.tracker = tracker
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.healthy_frames_to_deactivate = int(healthy_frames_to_deactivate)
        self.pixel_mask_decay = float(pixel_mask_decay)

        gx, gy = self.tracker.grid
        self.active_tiles = set()  # set of (j,i)
        self.tile_healthy_count = np.zeros((gy, gx), dtype=np.int32)

        self._persist_mask_float = None  # float [0..1]; thresholded to uint8 for inpaint

    # ---------- helpers ----------

    def _bbox_to_tile(self, bbox):
        (x0, y0, x1, y1) = bbox
        if self.tracker.shape is None:
            return (0, 0)
        h, w = self.tracker.shape
        gx, gy = self.tracker.grid
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)
        i = min(int(x0 // step_x), gx - 1)
        j = min(int(y0 // step_y), gy - 1)
        return (j, i)

    def activate_camera_tiles(self, boxes):
        for b in boxes:
            if b.get("kind") == "smudge" and b.get("why", {}).get("issue", "camera") == "camera":
                j, i = self._bbox_to_tile(b["bbox"])
                self.active_tiles.add((j, i))

    def deactivate_recovered_tiles(self):
        # Increment healthy counters for active tiles; deactivate if sustained healthy.
        gx, gy = self.tracker.grid
        scores = self.tracker.tile_scores if self.tracker.tile_scores is not None else np.zeros((gy, gx))
        newly_deactivated = []
        for (j, i) in list(self.active_tiles):
            if scores[j, i] < 0.15:  # considered "healthy"
                self.tile_healthy_count[j, i] += 1
            else:
                self.tile_healthy_count[j, i] = 0
            if self.tile_healthy_count[j, i] >= self.healthy_frames_to_deactivate:
                self.active_tiles.remove((j, i))
                newly_deactivated.append((j, i))
        return newly_deactivated

    def _apply_smudge_fix_tiles(self, frame_bgr, tiles_or_boxes):
        """Apply CLAHE + mild unsharp + gentle gamma (<1) to given tiles."""
        out = frame_bgr.copy()
        # Normalize to tiles
        tile_list = []
        if len(tiles_or_boxes) == 0:
            return out
        if isinstance(tiles_or_boxes[0], tuple):
            tile_list = tiles_or_boxes
        else:
            tile_list = [self._bbox_to_tile(b["bbox"]) for b in tiles_or_boxes]

        if self.tracker.shape is None:
            return out
        h, w = self.tracker.shape
        gx, gy = self.tracker.grid
        step_x = max(1, w // gx)
        step_y = max(1, h // gy)

        for (j, i) in tile_list:
            x0 = i * step_x; y0 = j * step_y
            x1 = w if i == gx - 1 else (i + 1) * step_x
            y1 = h if j == gy - 1 else (j + 1) * step_y

            roi = out[y0:y1, x0:x1]
            # LAB → CLAHE on L
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b_ = cv2.split(lab)
            l_eq = self._clahe.apply(l)

            # mild unsharp on L
            blur = cv2.GaussianBlur(l_eq, (0, 0), sigmaX=1.25)
            l_sharp = cv2.addWeighted(l_eq, 1.5, blur, -0.5, 0)

            # gentle gamma (<1 => brighten mid-tones)
            l_sharp = np.clip(l_sharp, 0, 255).astype(np.uint8)
            l_norm = l_sharp / 255.0
            gamma = 0.9
            l_gamma = np.power(l_norm, gamma)
            l_out = np.clip(l_gamma * 255.0, 0, 255).astype(np.uint8)

            lab_eq = cv2.merge([l_out, a, b_])
            fixed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            out[y0:y1, x0:x1] = fixed

        return out

    def _apply_pixel_inpaint(self, frame_bgr, mask_uint8):
        if mask_uint8 is None or np.count_nonzero(mask_uint8) == 0:
            return frame_bgr
        return cv2.inpaint(frame_bgr, mask_uint8, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    def _update_persistent_pixel_mask(self, new_mask_uint8):
        # Keep a float mask with decay in [0,1]; union with new detections.
        if new_mask_uint8 is None:
            return np.zeros((1, 1), dtype=np.uint8)
        m = (new_mask_uint8 > 0).astype(np.float32)
        if self._persist_mask_float is None or self._persist_mask_float.shape != m.shape:
            self._persist_mask_float = m.copy()
        else:
            self._persist_mask_float = np.maximum(self._persist_mask_float * self.pixel_mask_decay, m)
        # Threshold to binary for inpaint; keep light persistence
        out = (self._persist_mask_float >= 0.4).astype(np.uint8) * 255
        return out

    # ---------- main correction entry ----------

    def correct_frame(self, frame_bgr, min_score=None):
        if min_score is None:
            min_score = self.tracker.persistence_hi

        # 1) pixel mask: decay + always inpaint
        immediate_mask = self.tracker.get_persistent_mask(min_score=min_score)
        persist_mask = self._update_persistent_pixel_mask(immediate_mask)
        tmp = self._apply_pixel_inpaint(frame_bgr, persist_mask)

        # 2) auto-activate camera tiles from current boxes (doesn't depend on drift)
        boxes = self.tracker.get_persistent_boxes(min_score=min_score)
        cam_boxes = [b for b in boxes if b.get("kind") == "smudge" and b.get("why", {}).get("issue", "camera") == "camera"]
        self.activate_camera_tiles(cam_boxes)

        # 3) always apply corrections to active tiles (even if not detected this frame)
        tmp = self._apply_smudge_fix_tiles(tmp, list(self.active_tiles))

        # (optionally also strengthen current camera boxes in addition to active set)
        if cam_boxes:
            tmp = self._apply_smudge_fix_tiles(tmp, cam_boxes)

        # 4) auto-deactivate recovered tiles
        self.deactivate_recovered_tiles()

        meta = {
            "boxes": boxes,  # include why.issue & issue_conf
            "mask_nonzero": int(np.count_nonzero(persist_mask)),
            "health": self.tracker.camera_health(),
            "active_tiles": int(len(self.active_tiles))
        }
        return tmp, meta

# ============================================================
# CameraEffectsModel (ResNet18 backbone)
# ============================================================

class CameraEffectsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feat)

    def extract_features(self, x):
        return self.backbone(x).view(x.size(0), -1)

    def extract_intermediate(self, x, layer='layer1'):
        modules = dict(self.backbone.named_children())
        out = x
        for name, block in modules.items():
            out = block(out)
            if name == layer:
                return out
        return out

# ============================================================
# Legacy BlurDetector (kept for compatibility)
# ============================================================

class BlurDetector:
    def __init__(self, blur_thresh=50, brightness_thresh=50, contrast_thresh=15, edge_thresh=0.02, min_area=500):
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
        edges = cv2.Canny(region, 100, 200)
        edge_density = np.sum(edges > 0) / (region.size + 1e-5)
        is_blurry = lap_var < self.blur_thresh
        is_low_contrast = contrast < self.contrast_thresh
        is_low_edges = edge_density < self.edge_thresh
        is_very_dark = brightness < self.brightness_thresh
        cause = None
        if is_blurry and (is_low_contrast or is_low_edges):
            if is_very_dark: cause = "low_light"
            elif is_low_edges: cause = "low_edges"
            elif is_low_contrast: cause = "low_contrast"
            else: cause = "flat"
        return lap_var, cause

# ============================================================
# Drift pieces (kept; do not gate corrections)
# ============================================================

class DriftLogger:
    def __init__(self, log_dir="drift_logs", cooldown=5.0, max_history=6):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.last_log_time = 0
        self.history = deque(maxlen=max_history)

    def log(self, image, distance, blindspots, feature=None):
        now = time.time()
        if now - self.last_log_time < self.cooldown:
            return False
        if self.history:
            last = self.history[-1]
            last_img = cv2.imread(last["img_path"])
            if last_img is not None:
                diff = np.mean(cv2.absdiff(last_img, image))
                if diff < 10:
                    # too similar; skip log
                    return False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"drift_{timestamp}_d{distance:.2f}"
        img_path = self.log_dir / f"{filename_base}.jpg"
        meta_path = self.log_dir / f"{filename_base}.json"
        cv2.imwrite(str(img_path), image)
        metadata = {"timestamp": timestamp, "distance": distance, "blindspots": blindspots}
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.history.append({
            "img_path": str(img_path),
            "meta_path": str(meta_path),
            "features": feature.cpu() if feature is not None else None
        })
        self.last_log_time = now
        return True

    def get_recent_logs(self):
        return list(self.history)

class BaselineUpdater:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def update_baseline_from_logs(self, logger):
        logs = logger.get_recent_logs()
        if len(logs) < 6:
            return None
        features_list = []
        for entry in logs:
            img = cv2.imread(entry["img_path"])
            if img is None: continue
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.extract_features(input_tensor)
                features_list.append(feat)
        if not features_list:
            return None
        new_baseline = torch.mean(torch.cat(features_list, dim=0), dim=0, keepdim=True)
        return new_baseline

class DriftDetector:
    def __init__(self, baseline, threshold=3.0):
        self.baseline = baseline
        self.threshold = threshold

    def detect(self, current_features):
        distance = torch.norm(current_features - self.baseline, dim=1).item()
        is_drift = distance > self.threshold
        return is_drift, distance

    def set_baseline(self, new_baseline):
        self.baseline = new_baseline

# ============================================================
# Camera Monitor (public API)
# ============================================================

class CameraMonitor:
    """
    API:
      • process_frame(frame) -> (is_drift, distance, boxes)
      • preprocess_for_yolo(frame, min_score=None) -> (corrected, meta)
        - meta["boxes"] with why.issue & issue_conf
        - meta["health"] = {area_masked_pct, tiles_flagged, needs_reoptimize}
        - meta["active_tiles"] = count
      • persistence helpers:
        - acknowledge_current_tiles(boxes)
        - label_current_as_setting(boxes)
        - maybe_unfreeze_recovered(strong=True)
        - persist_camera_corrections(boxes)
        - unstick_recovered()
    """
    def __init__(self,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 grid=(8,6),
                 persistence_hi=0.85,
                 ratio_lap=0.55, ratio_edge=0.55, ratio_contr=0.65,
                 global_min_edges=0.01, global_min_lap=30.0,
                 freeze_margin=0.85,
                 seq_len=30,
                 pixel_mask_decay=0.98,
                 healthy_frames_to_deactivate=30):
        self.device = device

        # Safe checkpoint load (optional file)
        self.model = None
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        try:
            ckpt = torch.load("camera_effects.pth", map_location=device, weights_only=True)
        except TypeError:
            try:
                ckpt = torch.load("camera_effects.pth", map_location=device)
            except Exception:
                ckpt = None
        except Exception:
            ckpt = None

        if ckpt and "model_state" in ckpt and "idx2label" in ckpt:
            num_classes = len(ckpt["idx2label"])
            self.model = CameraEffectsModel(num_classes).to(device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
        else:
            # minimal backbone for features only (no classifier use)
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
            self.model = nn.Sequential(*list(base.children())[:-1])

        # Tracker + Corrector
        self.blindspot_tracker = BlindspotTracker(
            grid=grid,
            persistence_hi=persistence_hi,
            ratio_lap_thresh=ratio_lap,
            ratio_edge_thresh=ratio_edge,
            ratio_contrast_thresh=ratio_contr,
            global_min_edges=global_min_edges,
            global_min_lap=global_min_lap,
            freeze_margin=freeze_margin,
            seq_len=seq_len
        )
        self.blindspot_corrector = BlindspotCorrector(
            tracker=self.blindspot_tracker,
            healthy_frames_to_deactivate=healthy_frames_to_deactivate,
            pixel_mask_decay=pixel_mask_decay
        )

        # Legacy blur (optional)
        self.blur_detector = BlurDetector()

        # Drift/baseline (does NOT gate corrections)
        self.logger = DriftLogger()
        self.updater = BaselineUpdater(self, device)  # not used directly; kept for API compatibility
        self.baseline = torch.zeros((1, 512)).to(device)
        self.drift_detector = DriftDetector(self.baseline)

    # For BaselineUpdater compatibility: expose extract_features
    def extract_features(self, x):
        with torch.no_grad():
            if isinstance(self.model, CameraEffectsModel):
                return self.model.extract_features(x)
            return self.model(x).view(x.size(0), -1)

    # ---------- public methods ----------

    def process_frame(self, frame):
        """
        Returns (is_drift, distance, boxes) — boxes from tracker with why.issue & issue_conf.
        """
        boxes, _ = self.blindspot_tracker.update(frame)

        # ResNet18 features
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.extract_features(img_tensor)  # (1,512)
        is_drift, distance = self.drift_detector.detect(features)

        # Optional logging on drift (does not gate)
        if boxes and is_drift:
            self.logger.log(frame, distance, boxes, feature=features)

        return is_drift, distance, boxes

    def preprocess_for_yolo(self, frame, min_score=None):
        # Ensure tracker state is updated on this frame, then correct.
        self.blindspot_tracker.update(frame)
        corrected, meta = self.blindspot_corrector.correct_frame(frame, min_score=min_score)
        return corrected, meta

    # ---------- persistence helpers for hotkeys/UI ----------

    def acknowledge_current_tiles(self, boxes):
        """Freeze + label as CAMERA (used when a human cleaned lens etc.)."""
        self.blindspot_tracker.acknowledge_tiles(boxes)

    def label_current_as_setting(self, boxes):
        """Label selected tiles as SETTING (scene/lighting)."""
        self.blindspot_tracker.label_setting_issue(boxes)

    def maybe_unfreeze_recovered(self, strong=True):
        """Unfreeze tiles that have stayed healthy for a while."""
        self.blindspot_tracker.clear_freeze_if_recovered(strong=strong)

    def persist_camera_corrections(self, boxes):
        """Activate persistent corrections + acknowledge (camera) in one go."""
        # Activate in corrector:
        cam_boxes = [b for b in boxes if b.get("kind") == "smudge" and b.get("why", {}).get("issue", "camera") == "camera"]
        self.blindspot_corrector.activate_camera_tiles(cam_boxes)
        # Freeze in tracker + add LSTM labels:
        self.blindspot_tracker.acknowledge_tiles(cam_boxes)

    def unstick_recovered(self):
        """Run deactivation pass immediately (auto-deactivate recovered tiles)."""
        self.blindspot_corrector.deactivate_recovered_tiles()
