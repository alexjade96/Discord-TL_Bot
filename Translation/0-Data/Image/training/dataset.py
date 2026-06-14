"""Build an LMDB dataset from collected images for EasyOCR fine-tuning.

The output format matches deep-text-recognition-benchmark, which EasyOCR's
trainer uses internally. Each LMDB entry stores raw PNG bytes keyed as
`image-{i}` and the text label keyed as `label-{i}`.

Usage:
    python dataset.py                              # build from default data/
    python dataset.py --split 0.9                 # 90/10 train/val split
    python dataset.py --augment --aug-factor 4    # 4 augmented copies per original
    python dataset.py --augment --aug-ops noise,jpeg,brightness
    python dataset.py --data ../data --out lmdb_train
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
LABELS_FILE = DATA_DIR / "labels.jsonl"
DEFAULT_OUT = Path(__file__).parent / "lmdb_train"

# All available augmentation operation names
ALL_OPS: list[str] = ["brightness", "contrast", "noise", "blur", "jpeg", "rotate"]


# ---------------------------------------------------------------------------
# Augmentation primitives — each accepts and returns a BGR uint8 ndarray
# ---------------------------------------------------------------------------

def _aug_brightness(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random brightness shift ±30 (simulate monitor calibration variation)."""
    shift = rng.randint(-30, 30)
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def _aug_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random contrast scale 0.8–1.2 around the image mean."""
    scale = rng.uniform(0.8, 1.2)
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * scale + mean, 0, 255).astype(np.uint8)


def _aug_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gaussian noise σ 5–15 (simulates JPEG artifact grain)."""
    sigma = rng.uniform(5, 15)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _aug_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light Gaussian blur with 3×3 or 5×5 kernel (compression softness)."""
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def _aug_jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Re-encode as JPEG at quality 60–85 then decode (Discord CDN recompresses)."""
    quality = rng.randint(60, 85)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def _aug_rotate(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Tiny rotation ±2° (slightly tilted screenshots), fill with white."""
    angle = rng.uniform(-2.0, 2.0)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


_OP_FN = {
    "brightness": _aug_brightness,
    "contrast":   _aug_contrast,
    "noise":      _aug_noise,
    "blur":       _aug_blur,
    "jpeg":       _aug_jpeg,
    "rotate":     _aug_rotate,
}


def augment_image(img: np.ndarray, ops: list[str], rng: random.Random) -> np.ndarray:
    """Apply a random non-empty subset of `ops` to `img`.

    At least one op is always applied. Each enabled op is independently
    included with 50% probability, then at least one is forced if none fired.
    """
    chosen = [op for op in ops if rng.random() < 0.5]
    if not chosen:
        chosen = [rng.choice(ops)]
    for op in chosen:
        img = _OP_FN[op](img, rng)
    return img


def _img_to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Core LMDB builder
# ---------------------------------------------------------------------------

def _require_lmdb():
    try:
        import lmdb
        return lmdb
    except ImportError:
        print("lmdb is required: pip install lmdb")
        sys.exit(1)


def load_entries(labels_file: Path) -> list[dict]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    with labels_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_lmdb(
    entries: list[dict],
    images_dir: Path,
    output_path: Path,
    map_size_gb: int = 4,
    aug_factor: int = 0,
    aug_ops: list[str] | None = None,
    seed: int = 42,
) -> int:
    """Write entries to an LMDB database. Returns the number of entries written.

    Args:
        aug_factor: Number of augmented copies to generate per original image.
                    0 (default) disables augmentation entirely.
        aug_ops:    Which augmentation operations to use. Defaults to ALL_OPS.
    """
    lmdb = _require_lmdb()
    rng = random.Random(seed)

    if aug_factor > 0 and aug_ops is None:
        aug_ops = ALL_OPS

    output_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=map_size_gb * 1024**3)

    written = 0
    corrected = 0
    with env.begin(write=True) as txn:
        for entry in entries:
            img_path = images_dir / entry["filename"]
            if not img_path.exists():
                print(f"  [skip] missing image: {entry['filename']}")
                continue

            # Prefer manually corrected text over raw OCR output
            raw_label = entry.get("correct_text") or entry.get("ocr_text") or ""
            label = raw_label.strip()
            if not label:
                continue
            if entry.get("correct_text"):
                corrected += 1

            # Original image — stored as-is from disk
            img_bytes = img_path.read_bytes()
            written += 1
            txn.put(f"image-{written}".encode(), img_bytes)
            txn.put(f"label-{written}".encode(), label.encode("utf-8"))

            # Augmented copies — decoded, transformed, re-encoded
            if aug_factor > 0:
                raw = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    continue
                for _ in range(aug_factor):
                    aug = augment_image(raw.copy(), aug_ops, rng)
                    written += 1
                    txn.put(f"image-{written}".encode(), _img_to_png_bytes(aug))
                    txn.put(f"label-{written}".encode(), label.encode("utf-8"))

        txn.put(b"num-samples", str(written).encode())

    env.close()
    if corrected:
        print(f"  Used manual corrections for {corrected}/{len(entries)} entries")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build LMDB training dataset.")
    parser.add_argument("--data", default=str(DATA_DIR), help="Path to data/ directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output LMDB path")
    parser.add_argument("--split", type=float, default=None,
                        help="Train/val split ratio (e.g. 0.9 → 90%% train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true",
                        help="Enable image augmentation to increase dataset variety")
    parser.add_argument("--aug-factor", type=int, default=3,
                        help="Augmented copies per original when --augment is set (default: 3)")
    parser.add_argument("--aug-ops", default=",".join(ALL_OPS),
                        help=f"Comma-separated augmentation ops (default: all). "
                             f"Available: {', '.join(ALL_OPS)}")
    args = parser.parse_args()

    data_dir = Path(args.data)
    images_dir = data_dir
    labels_file = data_dir / "labels.jsonl"

    aug_factor = 0
    aug_ops: list[str] | None = None
    if args.augment:
        aug_factor = args.aug_factor
        aug_ops = [op.strip() for op in args.aug_ops.split(",") if op.strip() in ALL_OPS]
        unknown = [op.strip() for op in args.aug_ops.split(",") if op.strip() not in ALL_OPS and op.strip()]
        if unknown:
            print(f"  [warn] unknown aug-ops ignored: {unknown}")
        if not aug_ops:
            print("  [error] no valid aug-ops — disabling augmentation")
            aug_factor = 0
        else:
            print(f"  Augmentation: {aug_factor}× per original | ops: {aug_ops}")

    print(f"Loading labels from {labels_file}")
    entries = load_entries(labels_file)
    print(f"  Found {len(entries)} entries")

    if args.split:
        random.seed(args.seed)
        random.shuffle(entries)
        split_idx = int(len(entries) * args.split)
        train_entries = entries[:split_idx]
        val_entries = entries[split_idx:]

        out = Path(args.out)
        train_out = out.parent / f"{out.name}_train"
        val_out = out.parent / f"{out.name}_val"

        n = build_lmdb(train_entries, images_dir, train_out,
                       aug_factor=aug_factor, aug_ops=aug_ops, seed=args.seed)
        print(f"  Train LMDB: {n} entries → {train_out}")
        n = build_lmdb(val_entries, images_dir, val_out, seed=args.seed)
        print(f"  Val LMDB:   {n} entries → {val_out}")
    else:
        n = build_lmdb(entries, images_dir, Path(args.out),
                       aug_factor=aug_factor, aug_ops=aug_ops, seed=args.seed)
        print(f"  LMDB: {n} entries → {args.out}")


if __name__ == "__main__":
    main()
