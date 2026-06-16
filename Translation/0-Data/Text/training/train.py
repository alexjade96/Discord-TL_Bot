"""Additive seq2seq fine-tuning for TL-Bot translation models.

Fine-tunes Helsinki-NLP/opus-mt models on collected Discord translation pairs.
'Additive' means each run resumes from the latest checkpoint if one exists,
otherwise initialises from HuggingFace pretrained weights.

Two model directions are supported:
  - mul-en  : opus-mt-mul-en  (any language → English)
  - en-{tgt}: opus-mt-en-{tgt} (English → target language, one model per language)

Requires: transformers, torch, datasets  (pip install transformers torch datasets)

Workflow:
    1. python dataset.py --split 0.9       # build train/val JSONL
    2. python train.py --direction mul-en  # fine-tune → English
    3. python train.py --direction en-ko   # fine-tune English → Korean
    4. python deploy.py                    # install to ~/.tl-bot/models/

Usage:
    python train.py --direction mul-en
    python train.py --direction en-ko --epochs 5 --batch 16
    python train.py --direction mul-en --train dataset/mul-en_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATASET_DIR = TRAINING_DIR / "dataset"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

# Maps direction name → HuggingFace base model
_BASE_MODELS = {
    "mul-en": "Helsinki-NLP/opus-mt-mul-en",
}


def _base_model(direction: str) -> str:
    if direction in _BASE_MODELS:
        return _BASE_MODELS[direction]
    # en-{tgt} → opus-mt-en-{tgt}
    if direction.startswith("en-"):
        tgt = direction[3:]
        return f"Helsinki-NLP/opus-mt-en-{tgt}"
    raise ValueError(f"Unknown direction: {direction!r}. Expected 'mul-en' or 'en-<lang>'.")


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"{pkg} is required: pip install {pkg}")
        sys.exit(1)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nRun dataset.py first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TranslationDataset:
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 128):
        torch = _require("torch")
        self._torch = torch
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        src = pair["source"]
        tgt = pair["target"]

        model_inputs = self.tokenizer(
            src,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        # Replace padding token id with -100 so loss ignores padding
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def train(args: argparse.Namespace) -> None:
    torch = _require("torch")
    transformers = _require("transformers")
    from transformers import MarianMTModel, MarianTokenizer

    direction = args.direction
    base_model = _base_model(direction)
    ckpt_dir = CHECKPOINT_DIR / direction

    # Resolve dataset paths
    train_path = Path(args.train) if args.train else DATASET_DIR / f"{direction}_train.jsonl"
    val_path = Path(args.val) if args.val else DATASET_DIR / f"{direction}_val.jsonl"

    print(f"\nDirection : {direction}")
    print(f"Base model: {base_model}")
    print(f"Train data: {train_path}")
    print(f"Val data  : {val_path}")
    print(f"Checkpoint: {ckpt_dir}")

    # Load tokenizer and model
    latest_ckpt = ckpt_dir / "latest"
    if latest_ckpt.exists():
        print(f"\nResuming from checkpoint: {latest_ckpt}")
        tokenizer = MarianTokenizer.from_pretrained(str(latest_ckpt))
        model = MarianMTModel.from_pretrained(str(latest_ckpt))
    else:
        print(f"\nNo checkpoint found — initialising from {base_model}")
        tokenizer = MarianTokenizer.from_pretrained(base_model)
        model = MarianMTModel.from_pretrained(base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}\n")
    model = model.to(device)

    # Load datasets
    train_pairs = load_jsonl(train_path)
    val_pairs = load_jsonl(val_path) if val_path.exists() else []
    print(f"Train: {len(train_pairs)} pairs")
    if val_pairs:
        print(f"Val  : {len(val_pairs)} pairs")

    train_ds = TranslationDataset(train_pairs, tokenizer, max_length=args.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
    )

    val_loader = None
    if val_pairs:
        val_ds = TranslationDataset(val_pairs, tokenizer, max_length=args.max_length)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch, num_workers=0,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        val_msg = ""

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            avg_val = val_loss / len(val_loader)
            val_msg = f" | val_loss={avg_val:.4f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_dir = ckpt_dir / "best"
                model.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_train:.4f}{val_msg} | {elapsed:.1f}s")

        # Always save latest for additive continuation
        model.save_pretrained(str(ckpt_dir / "latest"))
        tokenizer.save_pretrained(str(ckpt_dir / "latest"))

    print(f"\nTraining complete. Checkpoints saved to {ckpt_dir}")
    print("Run `python deploy.py` to install the model for inference.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TL-Bot translation model.")
    parser.add_argument(
        "--direction", required=True,
        help="Model direction: 'mul-en' or 'en-<lang>' (e.g. en-ko, en-zh)",
    )
    parser.add_argument("--train", default=None, help="Path to train JSONL (default: dataset/<dir>_train.jsonl)")
    parser.add_argument("--val", default=None, help="Path to val JSONL (default: dataset/<dir>_val.jsonl)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128, dest="max_length")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
