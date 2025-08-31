import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import numpy as np
import cv2
import os
from glob import glob
import csv


class FullImageStatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        for p in self.resnet.parameters():
            p.requires_grad = False

    def extract_deep_features(self, image_tensor):
        with torch.no_grad():
            features = self.resnet(image_tensor).squeeze().flatten()
        return features.cpu().numpy()


def extract_visual_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_mean = img.mean(axis=(0, 1))
    rgb_std = img.std(axis=(0, 1))
    brightness = np.mean(gray)
    contrast = gray.std()
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = -np.sum((p := np.histogram(gray, bins=256)[0] / (gray.size + 1e-8)) * np.log2(p + 1e-8))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edge_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    gamma = np.mean(np.power(img / 255.0 + 1e-8, 2.2))
    skew = ((gray - gray.mean())**3).mean() / (gray.std()**3 + 1e-8)
    kurtosis = ((gray - gray.mean())**4).mean() / (gray.std()**4 + 1e-8)
    return np.array(list(rgb_mean) + list(rgb_std) + [brightness, contrast, lap_var, entropy, edge_mag, gamma, skew, kurtosis], dtype=np.float32)


def extract_and_save_to_csv(image_folder="calibrator_refs", output_csv="all_reference_features.csv"):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    image_paths = glob(os.path.join(image_folder, "**", "*.jpg"), recursive=True)
    extractor = FullImageStatExtractor()
    extractor.eval()

    stat_names = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
                  "brightness", "contrast", "lap_var", "entropy", "edge_mag",
                  "gamma", "skew", "kurtosis"]
    deep_names = [f"deep_{i}" for i in range(512)]
    columns = ["filepath"] + stat_names + deep_names

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for path in image_paths:
            img = cv2.imread(path)
            if img is None or img.shape[0] < 64 or img.shape[1] < 64:
                continue
            try:
                stats = extract_visual_stats(img)
                img_tensor = transform(img).unsqueeze(0)
                deep_feat = extractor.extract_deep_features(img_tensor)
                combined = np.concatenate([stats, deep_feat])
                writer.writerow([path] + combined.tolist())
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                continue

    print(f"Saved all features to {output_csv}")


if __name__ == "__main__":
    extract_and_save_to_csv()
