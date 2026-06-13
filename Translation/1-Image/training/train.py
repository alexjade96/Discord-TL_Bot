"""Additive fine-tuning for the TL-Bot OCR recognition model.

Uses the None-VGG-BiLSTM-CTC architecture from tl_bot_ocr.py, trained on
the dataset built by dataset.py. 'Additive' means each run continues from
the latest checkpoint (checkpoints/tl_bot_ocr_latest.pth) if one exists,
otherwise starts from EasyOCR's pretrained English weights.

Workflow:
    1. python dataset.py --split 0.9      # build train + val LMDB
    2. python train.py                    # train (or continue)
    3. python deploy.py                   # install into ~/.EasyOCR/

Usage:
    python train.py
    python train.py --epochs 20 --batch 64
    python train.py --train lmdb_train_train --val lmdb_train_val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).parent
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
DEFAULT_TRAIN_LMDB = TRAINING_DIR / "lmdb_train_train"
DEFAULT_VAL_LMDB = TRAINING_DIR / "lmdb_train_val"

# Character set must match tl_bot_ocr.yaml
CHARACTERS = (
    "0123456789abcdefghijklmnopqrstuvwxyz"
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
)
NUM_CLASS = len(CHARACTERS) + 1  # +1 for CTC blank


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"{pkg} is required: pip install {pkg}")
        sys.exit(1)


class LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: Path, img_h: int = 32, img_w: int = 100):
        lmdb = _require("lmdb")
        import numpy as np, cv2  # noqa: E401

        self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = int(txn.get(b"num-samples").decode())
        self.img_h = img_h
        self.img_w = img_w
        self._np = np
        self._cv2 = cv2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np, cv2 = self._np, self._cv2
        with self.env.begin() as txn:
            img_bytes = txn.get(f"image-{idx + 1}".encode())
            label = txn.get(f"label-{idx + 1}".encode()).decode("utf-8")

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        # Encode label as indices into CHARACTERS; unknown chars → blank (0)
        char_to_idx = {c: i + 1 for i, c in enumerate(CHARACTERS)}
        encoded = [char_to_idx.get(c, 0) for c in label]
        return img, torch.tensor(encoded, dtype=torch.long), len(encoded)


def _collate(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    max_len = max(lengths)
    padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, (lbl, ln) in enumerate(zip(labels, lengths)):
        padded[i, :ln] = lbl
    return images, padded, torch.tensor(lengths, dtype=torch.long)


def _load_model(checkpoint_dir: Path, device: torch.device):
    sys.path.insert(0, str(TRAINING_DIR))
    from tl_bot_ocr import Model

    model = Model(
        input_channel=1,
        output_channel=256,
        hidden_size=256,
        num_class=NUM_CLASS,
    ).to(device)

    latest = checkpoint_dir / "tl_bot_ocr_latest.pth"
    if latest.exists():
        print(f"Resuming from checkpoint: {latest}")
        state = torch.load(str(latest), map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        print("No checkpoint found — starting from random initialisation.")
        print("Tip: copy EasyOCR's pretrained english_g2.pth here as tl_bot_ocr_latest.pth")
        print("     to start from a stronger baseline.")

    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train)
    val_path = Path(args.val) if args.val else None

    if not train_path.exists():
        print(f"Training LMDB not found: {train_path}")
        print("Run `python dataset.py --split 0.9` first.")
        sys.exit(1)

    train_ds = LmdbDataset(train_path)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    print(f"Train: {len(train_ds)} samples")

    val_loader = None
    if val_path and val_path.exists():
        val_ds = LmdbDataset(val_path)
        val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=_collate, num_workers=0)
        print(f"Val:   {len(val_ds)} samples")

    model = _load_model(CHECKPOINT_DIR, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels, lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)  # (B, T, num_class)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C) for CTC
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)

            loss = ctc_loss(log_probs, labels, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        val_msg = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels, lengths in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    log_probs = logits.log_softmax(2).permute(1, 0, 2)
                    input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
                    val_loss += ctc_loss(log_probs, labels, input_lengths, lengths).item()
            avg_val = val_loss / len(val_loader)
            val_msg = f" | val_loss={avg_val:.4f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = CHECKPOINT_DIR / "tl_bot_ocr_best.pth"
                torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_train:.4f}{val_msg} | {elapsed:.1f}s")

        # Always save latest for additive continuation
        torch.save(model.state_dict(), CHECKPOINT_DIR / "tl_bot_ocr_latest.pth")

    print(f"\nTraining complete. Checkpoints saved to {CHECKPOINT_DIR}")
    print("Run `python deploy.py` to install the model for EasyOCR inference.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TL-Bot OCR model.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_LMDB))
    parser.add_argument("--val", default=str(DEFAULT_VAL_LMDB) if DEFAULT_VAL_LMDB.exists() else None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
