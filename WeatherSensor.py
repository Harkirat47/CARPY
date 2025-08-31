import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18

import cv2
import numpy as np

class WeatherClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ImageEnhancementNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class WeatherSensor:
    def __init__(self, device='cpu'):
        self.device = device
        self.weather_classes = ['sunny', 'cloudy', 'foggy', 'misty', 'rainy', 
                                'stormy', 'snowy', 'lowlight', 'dusty']
        
        self.weather_classifier = WeatherClassifier(len(self.weather_classes)).to(device)
        self.enhancer = ImageEnhancementNN().to(device)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.weather_history = []
        self.history_length = 5
        self.frame_skip = 0
        
    def extract_weather_features(self, frame):
        small_frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = gray.std()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return np.array([brightness, contrast, edge_density, lap_var])

    def classify_weather(self, frame):
        self.frame_skip += 1
        if self.frame_skip % 3 != 0 and len(self.weather_history) > 0:
            return self.weather_history[-1], 0.8

        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.weather_classifier(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        weather_condition = self.weather_classes[pred_idx]
        features = self.extract_weather_features(frame)

        if features[0] < 50:
            weather_condition = 'lowlight'
        elif features[2] < 0.1 and features[3] < 100:
            weather_condition = 'foggy'
        
        self.weather_history.append(weather_condition)
        if len(self.weather_history) > self.history_length:
            self.weather_history.pop(0)
        
        stable_weather = self.weather_history[-1] if self.weather_history else weather_condition
        return stable_weather, confidence

    def correct_sunny_conditions(self, frame):
        gamma = 0.8
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def correct_cloudy_conditions(self, frame):
        return cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    def correct_foggy_conditions(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return cv2.convertScaleAbs(result, alpha=1.3, beta=10)

    def correct_misty_conditions(self, frame):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)

    def correct_rainy_conditions(self, frame):
        enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def correct_stormy_conditions(self, frame):
        return cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    def correct_snowy_conditions(self, frame):
        gamma = 0.7
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def correct_lowlight_conditions(self, frame):
        gamma = 0.5
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        brightened = cv2.LUT(frame, table)

        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        final = cv2.convertScaleAbs(denoised, alpha=1.2, beta=15)
        return final

    def correct_dusty_conditions(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)

    def neural_enhancement(self, frame):
        if self.frame_skip % 2 != 0:
            return frame

        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // 2, h // 2))

        frame_tensor = torch.from_numpy(small_frame.transpose(2, 0, 1)).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            enhanced = self.enhancer(frame_tensor)
            enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced = (enhanced * 255).astype(np.uint8)

        return cv2.resize(enhanced, (w, h))

    def apply_corrections(self, frame, weather_condition):
        correction_map = {
            'sunny': self.correct_sunny_conditions,
            'cloudy': self.correct_cloudy_conditions,
            'foggy': self.correct_foggy_conditions,
            'misty': self.correct_misty_conditions,
            'rainy': self.correct_rainy_conditions,
            'stormy': self.correct_stormy_conditions,
            'snowy': self.correct_snowy_conditions,
            'lowlight': self.correct_lowlight_conditions,
            'dusty': self.correct_dusty_conditions
        }
        corrected = correction_map.get(weather_condition, lambda x: x)(frame)
        return self.neural_enhancement(corrected)

    def process_frame(self, frame):
        weather_condition, confidence = self.classify_weather(frame)
        corrected_frame = self.apply_corrections(frame, weather_condition)
        return corrected_frame, weather_condition, confidence
