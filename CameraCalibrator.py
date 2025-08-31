import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

class CameraConditionNN(nn.Module):
    def __init__(self, input_dim=64, output_dim=30):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

class CameraCalibrator:
    def __init__(self, model_path=None, device=None):
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Calibrator] using device: {self.device}")

        # Load small MLP
        self.model = CameraConditionNN().to(self.device)
        if model_path and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ResNet-18 backbone (drop final classifier)
        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Image transforms for ResNet
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

        # Load & precompute reference patches
        self.reference_images = self._load_reference_samples()
        self.ref_data = self._precompute_reference_data()

    def _load_reference_samples(self):
        refs = {}
        for cls in [0, 2, 16]:
            folder = os.path.join("calibrator_refs", str(cls))
            os.makedirs(folder, exist_ok=True)
            imgs = []
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    img = cv2.imread(os.path.join(folder, f))
                    if img is not None:
                        imgs.append(img)
            if not imgs:
                print(f"[Warning] no refs for class {cls}, using blank patch")
                imgs = [np.zeros((64,64,3), dtype=np.uint8)]
            refs[cls] = imgs
        return refs

    def _visual_stats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_mean = img.mean(axis=(0,1))
        rgb_std  = img.std(axis=(0,1))
        brightness = gray.mean()
        contrast   = gray.std()
        lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx     = cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2
        sobely     = cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2
        tenengrad  = np.mean(sobelx + sobely)
        fft_blur   = np.mean(np.abs(np.fft.fft2(gray)))
        hist       = np.histogram(gray, bins=256)[0].astype(np.float32)
        p          = hist/(hist.sum()+1e-8)
        entropy    = -np.sum(p*np.log2(p+1e-8))
        edge_mag   = np.sqrt(sobelx + sobely).mean()
        block      = np.float32(gray[:32,:32]) / 255.0
        artifact   = np.mean(np.abs(cv2.dct(block)))
        mu, sigma  = gray.mean(), gray.std()+1e-8
        skew       = ((gray-mu)**3).mean()/(sigma**3)
        kurt       = ((gray-mu)**4).mean()/(sigma**4)
        smudge     = np.mean(np.var(img.reshape(-1,3),axis=0)<15)
        corners    = cv2.goodFeaturesToTrack(gray,100,0.01,10)
        corner_den = len(corners) if corners is not None else 0

        stats = [
            *rgb_mean.tolist(), *rgb_std.tolist(),
            brightness, contrast,
            lap_var, tenengrad, fft_blur,
            entropy, edge_mag, artifact,
            skew, kurt, smudge, corner_den
        ]
        return np.array(stats, dtype=np.float32), rgb_mean

    def _precompute_reference_data(self):
        data = {}
        for cls, imgs in self.reference_images.items():
            entries = []
            for img in imgs:
                stats, rgb = self._visual_stats(img)
                with torch.no_grad():
                    t    = self.transform(img).unsqueeze(0).to(self.device)
                    feat = self.feature_extractor(t).view(-1).cpu().numpy()
                entries.append((stats, rgb, feat))
            data[cls] = entries
        return data

    def _rgb_to_hex(self, rgb):
        r,g,b = [int(c) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _analyze_grid_blindspots(self, frame, grid_size=6):
        h,w  = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 5.0:
            return []
        gh,gw = h//grid_size, w//grid_size
        blind = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                if cv2.Laplacian(cell, cv2.CV_64F).var() < 6.0:
                    blind.append((i,j))
        return blind

    def predict(self, frame, detections):
        crops, refs, confs, rgbs = [], [], [], []
        for det in detections:
            cls = det["class"]
            if det["confidence"] < 0.7 or cls not in self.ref_data:
                continue
            x1,y1,x2,y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            stats_r, rgb_r, feat_r = random.choice(self.ref_data[cls])
            stats_c, rgb_c         = self._visual_stats(crop)
            diff_stats             = stats_c - stats_r

            crops.append(self.transform(crop))
            refs.append((feat_r, diff_stats))
            rgbs.append(rgb_c)
            confs.append(det["confidence"])

        if not crops:
            avg_feature = np.zeros(64, dtype=np.float32)
            avg_rgb     = np.zeros(3, dtype=np.float32)
        else:
            batch = torch.stack(crops).to(self.device)
            with torch.no_grad():
                feats = self.feature_extractor(batch).view(len(crops), -1).cpu().numpy()

            feats_list = []
            for i, (feat_r, diff_stats) in enumerate(refs):
                deep_diff = feats[i] - feat_r
                vec       = np.concatenate([diff_stats, deep_diff], axis=0)
                feats_list.append(vec[:64] * confs[i])

            avg_feature = np.mean(feats_list, axis=0)
            avg_rgb     = np.mean(rgbs, axis=0)

        inp = torch.tensor(avg_feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(inp).cpu().numpy()[0]

        blinds = self._analyze_grid_blindspots(frame, grid_size=6)
        return {
            "prediction":       pred,
            "tint_hex":         self._rgb_to_hex(avg_rgb),
            "avg_rgb":          avg_rgb.tolist(),
            "blind_grid_coords": blinds
        }
