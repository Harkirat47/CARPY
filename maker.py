import os
import cv2
import torch
import csv
import json
import numpy as np
import torchvision.transforms as T
from torchvision.models import resnet18
from urllib.request import urlretrieve
from tqdm import tqdm

# Settings
COCO_IMAGES_URL = "http://images.cocodataset.org/train2017/"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_PATH = "annotations/instances_train2017.json"
IMAGE_DIR = "calibrator_refs/train2017"
ALLOWED_CLASSES = {"person": 1, "car": 3, "bus": 6}  # COCO class ID mapping

# Feature Extractor
class FullImageStatExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
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

def download_coco_subset(annotation_path=ANNOTATION_PATH, out_dir=IMAGE_DIR, max_per_class=150):
    print("Parsing annotations...")
    with open(annotation_path, "r") as f:
        data = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    cls_counts = {cls: 0 for cls in ALLOWED_CLASSES.values()}
    selected = []

    for ann in data["annotations"]:
        cls_id = ann["category_id"]
        if cls_id in cls_counts and cls_counts[cls_id] < max_per_class:
            img_id = ann["image_id"]
            filename = img_id_to_file[img_id]
            if filename not in selected:
                selected.append(filename)
                cls_counts[cls_id] += 1
        if all(v >= max_per_class for v in cls_counts.values()):
            break

    os.makedirs(out_dir, exist_ok=True)
    print("Downloading images...")
    for fname in tqdm(selected):
        url = COCO_IMAGES_URL + fname
        local_path = os.path.join(out_dir, fname)
        if not os.path.exists(local_path):
            try:
                urlretrieve(url, local_path)
            except:
                continue

def extract_and_save_to_csv(image_folder=IMAGE_DIR, output_csv="coco_reference_features.csv"):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor()
    ])

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

        files = sorted(os.listdir(image_folder))
        for i, file in enumerate(tqdm(files)):
            path = os.path.join(image_folder, file)
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
                print(f"Skipping {file}: {e}")
                continue

    print(f"[DONE] Saved features to {output_csv}")

if __name__ == "__main__":
    if not os.path.exists(ANNOTATION_PATH):
        print("[ERROR] You must download and unzip COCO 2017 annotations manually first.")
        print("Run: wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("Then unzip and place 'annotations/instances_train2017.json' here.")
    else:
        download_coco_subset()
        extract_and_save_to_csv()
