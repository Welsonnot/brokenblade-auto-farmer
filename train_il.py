"""
Phase 1 - Behavioural Cloning / Imitation Learning
====================================================
Trains ILPolicyNet on every recording session in recordings/.
Handles the FULL game loop - navigation, boss lock, combat, waiting -
because the recorder captures every key you press regardless of phase.

Saves:  models/il_baseline.pth

Usage:
    python train_il.py
    python train_il.py --epochs 80 --lr 2e-4 --batch 64
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.policy_net import ACTION_KEYS, N_ACTIONS, ILPolicyNet

# -- Defaults ------------------------------------------------------------------
RECORDINGS_DIR = "recordings"
MODEL_OUT      = "models/il_baseline.pth"
FRAME_SIZE     = (224, 224)
LR             = 1e-4
EPOCHS         = 60
BATCH          = 32
VAL_SPLIT      = 0.10
PATIENCE       = 12       # early-stop if val doesn't improve for N epochs
GRAD_ACCUM     = 2        # effective batch = BATCH x GRAD_ACCUM
NUM_WORKERS    = 2

# CSV column -> ACTION_KEY mapping.
# The new universal recorder uses the same name in the CSV as in ACTION_KEYS,
# so this is an identity map.  Old recordings simply won't have columns for
# newer keys (space, shift, e, r, c, v, 1-5, m2) - those default to 0, which
# is correct since you weren't pressing them in those sessions.
_CSV_COL: dict[str, str] = {k: k for k in ACTION_KEYS}


# -- Dataset -------------------------------------------------------------------

class DemoDataset(Dataset):
    """
    Scans every subdirectory of RECORDINGS_DIR for  inputs.csv + frame_*.jpg.
    Supports both old (no Q) and new (with Q) recorder formats.
    """

    def __init__(self, recordings_dir: str = RECORDINGS_DIR) -> None:
        self.samples: list[tuple[str, np.ndarray]] = []
        self._load(recordings_dir)

    def _load(self, root: str) -> None:
        sessions = sorted(Path(root).glob("*/inputs.csv"))
        if not sessions:
            raise FileNotFoundError(
                f"No recording sessions found under '{root}/'.\n"
                "Run  python recorder.py  and record at least one session first."
            )
        print(f"[IL] Loading {len(sessions)} session(s) ...")
        for csv_path in sessions:
            try:
                self.samples.extend(self._parse_session(csv_path))
            except Exception as exc:
                print(f"  [WARN] Skipped {csv_path.parent.name}: {exc}")
        if not self.samples:
            raise RuntimeError("All sessions failed to load - check recordings/ folder.")
        print(f"[IL] {len(self.samples):,} total frames ready.")

    def _parse_session(self, csv_path: Path) -> list[tuple[str, np.ndarray]]:
        session_dir = csv_path.parent
        rows: list[tuple[str, np.ndarray]] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            cols   = set(reader.fieldnames or [])
            for row in reader:
                frame_path = session_dir / row['frame_file']
                if not frame_path.exists():
                    continue
                label = np.zeros(N_ACTIONS, dtype=np.float32)
                for idx, key in enumerate(ACTION_KEYS):
                    col = _CSV_COL[key]
                    if col in cols:
                        label[idx] = float(row[col])
                    # else: key absent (old recording) -> stays 0
                rows.append((str(frame_path), label))
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((*FRAME_SIZE[::-1], 3), dtype=np.uint8)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [0,1]
        return frame, torch.from_numpy(label)


def _pos_weights(dataset: DemoDataset, device: torch.device) -> torch.Tensor:
    """
    Per-action class-balance weights for BCEWithLogitsLoss.
    Upweights rare actions (e.g. Z/X which fire infrequently).
    """
    counts = np.zeros(N_ACTIONS, dtype=np.float64)
    total  = len(dataset)
    for _, label in dataset.samples:
        counts += label
    pw = (total - counts) / np.clip(counts, 1.0, None)
    pw = np.clip(pw, 1.0, 50.0)
    labels = [f"{k}:{v:.1f}" for k, v in zip(ACTION_KEYS, pw)]
    print(f"[IL] Pos weights  ->  {', '.join(labels)}")
    return torch.tensor(pw, dtype=torch.float32, device=device)


# -- Training loop -------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IL] Device: {device}")

    # Dataset
    full_ds  = DemoDataset(args.recordings)
    val_n    = max(1, int(len(full_ds) * VAL_SPLIT))
    trn_n    = len(full_ds) - val_n
    trn_ds, val_ds = random_split(
        full_ds, [trn_n, val_n],
        generator=torch.Generator().manual_seed(42))

    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[IL] Train: {trn_n:,}  |  Val: {val_n:,}  |  "
          f"Batch: {args.batch}  |  Epochs: {args.epochs}")

    # Model + loss
    model   = ILPolicyNet().to(device)
    pos_w   = _pos_weights(full_ds, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs("models", exist_ok=True)
    best_val  = float('inf')
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # -- Train ------------------------------------------------------------
        model.train()
        trn_loss = 0.0
        optimizer.zero_grad()

        for step, (frames, labels) in enumerate(trn_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(frames), labels) / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            trn_loss += loss.item() * GRAD_ACCUM

        trn_loss /= max(len(trn_loader), 1)

        # -- Validate ---------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for frames, labels in val_loader:
                val_loss += loss_fn(
                    model(frames.to(device, non_blocking=True)),
                    labels.to(device, non_blocking=True)
                ).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"trn={trn_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.1e}  {elapsed:.1f}s")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"           [OK] New best val={best_val:.4f}  ->  saved {MODEL_OUT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[IL] Early stop at epoch {epoch}  "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print(f"\n[IL] Training complete.  Best val loss: {best_val:.4f}")
    print(f"[IL] Model saved ->  {MODEL_OUT}")
    print(f"[IL] Next step   ->  python train_rl.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioural Cloning - Phase 1")
    parser.add_argument("--recordings", default=RECORDINGS_DIR,
                        help="Root folder containing recording sessions")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--batch",      type=int,   default=BATCH)
    train(parser.parse_args())
