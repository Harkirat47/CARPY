import cv2
import torch
import urllib.request
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# Load model
model_type = "DPT_Large"  # Use "MiDaS_small" for faster inference
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    input_tensor = transform(input_image).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

    # Show result
    cv2.imshow("Depth Map", depth_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
