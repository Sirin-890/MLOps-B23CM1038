import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import torch

def compute_metrics(preds, targets, num_classes=23):
    """
    preds: (B, H, W)
    targets: (B, H, W)
    """
    ious = []
    dices = []

    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue  # ignore class not present

        iou = intersection / union
        dice = (2 * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-6)

        ious.append(iou)
        dices.append(dice)

    miou = sum(ious) / len(ious) if ious else 0
    mdice = sum(dices) / len(dices) if dices else 0

    return miou, mdice
# =========================
# CONFIG
# =========================
IMG_DIR = "CameraRGB"
MASK_DIR = "CameraMask"

BATCH_SIZE = 8
SEED = 42
NUM_CLASSES = 23

# =========================
# DATASET CLASS
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        img_path = os.path.join(self.img_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        # Load
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to numpy
        image = np.array(image)
        mask = np.array(mask)

        # If mask is RGB → convert to single channel (optional fix)
        if len(mask.shape) == 3:
            # simple fallback (NOT perfect, but avoids crash)
            mask = mask[:, :, 0]

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# =========================
# TRAIN-TEST SPLIT
# =========================
files = sorted(os.listdir(IMG_DIR))

random.seed(SEED)
random.shuffle(files)

split_idx = int(0.8 * len(files))

train_files = files[:split_idx]
test_files = files[split_idx:]

print(f"Total samples: {len(files)}")
print(f"Train: {len(train_files)}, Test: {len(test_files)}")

# =========================
# DATASETS
# =========================
train_dataset = SegmentationDataset(IMG_DIR, MASK_DIR, train_files)
test_dataset = SegmentationDataset(IMG_DIR, MASK_DIR, test_files)

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# SANITY CHECK
# =========================
print("\nRunning sanity check...")

images, masks = next(iter(train_loader))

print("Image shape:", images.shape)   # (B, 3, H, W)
print("Mask shape:", masks.shape)     # (B, H, W)
print("Mask min:", masks.min().item())
print("Mask max:", masks.max().item())
import segmentation_models_pytorch as smp
import torch

NUM_CLASSES = 23

model = smp.Unet(
    encoder_name="resnet34",     
    encoder_weights=None,      
    in_channels=3,
    classes=NUM_CLASSES
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

import torch.nn as nn
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20

train_losses = []
val_ious = []
val_dices = []

for epoch in range(EPOCHS):
    # ================= TRAIN =================
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ================= VALIDATION =================
    model.eval()
    total_iou = 0
    total_dice = 0
    count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            miou, mdice = compute_metrics(preds, masks)

            total_iou += miou
            total_dice += mdice
            count += 1

    avg_iou = total_iou / count
    avg_dice = total_dice / count

    val_ious.append(avg_iou)
    val_dices.append(avg_dice)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f} | mIoU: {avg_iou:.4f} | mDice: {avg_dice:.4f}")

import torch
import os

os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/unet_model.pth")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.savefig("loss_curve.png")   # ✅ saves in current directory
plt.close()

plt.figure()
plt.plot(val_ious, label="mIoU")
plt.plot(val_dices, label="mDice")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.legend()

plt.savefig("metrics_curve.png")   # ✅ saves file
plt.close()