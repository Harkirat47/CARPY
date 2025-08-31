# train_camera_id.py
import os

# Kaggle creds here 
os.environ['KAGGLE_USERNAME'] = 'harkirathattar'
os.environ['KAGGLE_KEY']      = 'a507117e5333b6a88a30f29bf23d7f82'

import io
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class DresdenKaggleDataset(Dataset):
    def __init__(self,
                 dataset="micscodes/dresden-image-database",
                 transform=None):
        self.api = KaggleApi()
        self.api.authenticate()   # will now pick up your creds from os.environ

        files = self.api.dataset_list_files(dataset).files
        self.files = [f.name for f in files
                      if f.name.lower().endswith(".jpg")]
        self.dataset = dataset
        self.transform = transform

        models = sorted({fn.split("/")[0] for fn in self.files})
        self.label2idx = {m:i for i,m in enumerate(models)}
        self.idx2label = {i:m for m,i in self.label2idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        data = self.api.dataset_download_file(
            self.dataset, file_name=fn, path=None
        )
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[fn.split("/")[0]]
        return img, label

class CameraIdentifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = torchvision.models.resnet18(pretrained=True)
        self.backbone   = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feat)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_camera_id] Using device: {device}")

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    ds     = DresdenKaggleDataset(transform=transform)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    model     = CameraIdentifier(num_classes=len(ds.label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:2d} — loss: {total_loss/len(loader):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "idx2label":   ds.idx2label
    }, "camera_id.pth")
    print("[train_camera_id] Saved camera_id.pth")

if __name__ == "__main__":
    main()
