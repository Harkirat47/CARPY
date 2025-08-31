import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

class MedianFilterTransform:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        np_img = np.array(img)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        filtered_img = cv2.medianBlur(np_img, self.kernel_size)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_img)

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        self.final_t = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.final_A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        A = self.final_A(d5)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        t = self.final_t(u5)
        J = (x - A * (1 - t)) / (t + 1e-8)
        return J, t, A

def process_frame(model, frame, device, median_filter, transform):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_filtered = median_filter(image)
    input_tensor = transform(image_filtered).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor, _, _ = model(input_tensor)

    output_tensor = (output_tensor * 0.5) + 0.5
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_tensor = output_tensor.squeeze(0).cpu()
    output_array = output_tensor.permute(1, 2, 0).numpy()
    output_frame = cv2.cvtColor((output_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return output_frame

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Generator().to(device)
    model_path = "generator.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print(f"Please ensure your trained model is in the same directory and the filename is correct.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")

        median_filter = MedianFilterTransform(kernel_size=3)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cap = cv2.VideoCapture(1) 
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            defogged_frame = process_frame(model, frame, device, median_filter, transform)
            resized_original = cv2.resize(frame, (256, 256))
            combined_display = np.hstack([resized_original, defogged_frame])
            cv2.imshow('Original vs. Defogged - Press Q to Quit', combined_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()