"""
Train - Step 3a
===============
Trains a MobileNetV2 to predict which keys (Z/X/C/G + left-click)
to press given a 224x224 game frame.  Multi-label BCEWithLogitsLoss.

Usage:
  python train_model.py

Best checkpoint saved to models/game_bot.pth

Install:  pip install torch torchvision
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

from legacy.dataset import GameDataset, LABELS

RECORDINGS_DIR = "recordings"
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "game_bot.pth")

EPOCHS     = 25
BATCH_SIZE = 128     # RTX 3060 12 GB handles this comfortably with MobileNetV2
LR         = 1e-4
NUM_WORKERS = 4      # Ryzen 5600X (6c/12t) - 4 workers keeps the GPU fed
VAL_SPLIT  = 0.1
NUM_CLASSES = len(LABELS)   # 8  (m1, z, x, f, w, a, s, d)


def build_model() -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.last_channel, NUM_CLASSES),
    )
    return m


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = GameDataset(RECORDINGS_DIR)
    if len(ds) == 0:
        print("No data found - record some gameplay with recorder.py first.")
        return
    print(f"Samples: {len(ds)}")

    val_n   = max(1, int(len(ds) * VAL_SPLIT))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model     = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                v_loss += criterion(model(imgs), labels).item()

        t_loss /= len(train_dl)
        v_loss /= len(val_dl)
        scheduler.step()

        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), MODEL_PATH)
            marker = "  <- saved"

        print(f"Epoch {epoch:02d}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}{marker}")

    print(f"\nDone.  Best model: {MODEL_PATH}")


if __name__ == "__main__":
    # Required on Windows when NUM_WORKERS > 0
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Lets cuDNN auto-tune kernels for your 3060 - speeds up training ~10-20%
    torch.backends.cudnn.benchmark = True

    train()
