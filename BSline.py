import torch
import cv2
from torchvision import transforms

class BaselineUpdater:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def update_baseline_from_logs(self, logger):
        logs = logger.get_recent_logs()
        if len(logs) < 6:
            print(f"ℹ️ Not enough logs to update baseline ({len(logs)}/6)")
            return None

        features_list = []
        for entry in logs:
            img = cv2.imread(entry["img_path"])
            if img is None:
                continue

            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model.extract_features(input_tensor)
                features_list.append(feat)

        if not features_list:
            print("⚠️ No valid features extracted.")
            return None

        new_baseline = torch.mean(torch.cat(features_list, dim=0), dim=0, keepdim=True)
        print("✅ Updated baseline from recent drift logs.")
        return new_baseline  # Shape: [1, 512]
