"""
PyTorch Dataset - shared by train_model.py and run_bot.py.
Reads all session subdirs under recordings/ and builds a flat sample list.
"""

import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

LABELS = ['m1', 'z', 'x', 'f', 'w', 'a', 's', 'd']

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


class GameDataset(Dataset):
    def __init__(self, recordings_dir: str = "recordings",
                 transform=None):
        self.transform = transform or _TRANSFORM
        self.samples: list[tuple[str, list[int]]] = []

        for session in sorted(os.listdir(recordings_dir)):
            csv_path = os.path.join(recordings_dir, session, "inputs.csv")
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    fpath = os.path.join(recordings_dir, session, row['frame_file'])
                    if not os.path.isfile(fpath):
                        continue
                    label = [int(row[k]) for k in LABELS]
                    self.samples.append((fpath, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
