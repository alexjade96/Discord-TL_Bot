"""Build an LMDB dataset from collected images for EasyOCR fine-tuning.

The output format matches deep-text-recognition-benchmark, which EasyOCR's
trainer uses internally. Each LMDB entry stores raw PNG bytes keyed as
`image-{i}` and the text label keyed as `label-{i}`.

Usage:
    python dataset.py                   # build from default data/
    python dataset.py --data ../data --out lmdb_train
    python dataset.py --split 0.9      # 90% train / 10% val (creates two DBs)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.jsonl"
DEFAULT_OUT = Path(__file__).parent / "lmdb_train"


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
    map_size_gb: int = 2,
) -> int:
    """Write entries to an LMDB database. Returns the number of entries written."""
    lmdb = _require_lmdb()

    output_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=map_size_gb * 1024**3)

    written = 0
    with env.begin(write=True) as txn:
        for i, entry in enumerate(entries, 1):
            img_path = images_dir / entry["filename"]
            if not img_path.exists():
                print(f"  [skip] missing image: {entry['filename']}")
                continue

            label = entry["ocr_text"].strip()
            if not label:
                continue

            img_bytes = img_path.read_bytes()
            txn.put(f"image-{i}".encode(), img_bytes)
            txn.put(f"label-{i}".encode(), label.encode("utf-8"))
            written += 1

        txn.put(b"num-samples", str(written).encode())

    env.close()
    return written


def main():
    parser = argparse.ArgumentParser(description="Build LMDB training dataset.")
    parser.add_argument("--data", default=str(DATA_DIR), help="Path to data/ directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output LMDB path")
    parser.add_argument("--split", type=float, default=None,
                        help="Train/val split ratio (e.g. 0.9 → 90%% train)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    images_dir = data_dir / "images"
    labels_file = data_dir / "labels.jsonl"

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

        n = build_lmdb(train_entries, images_dir, train_out)
        print(f"  Train LMDB: {n} entries → {train_out}")
        n = build_lmdb(val_entries, images_dir, val_out)
        print(f"  Val LMDB:   {n} entries → {val_out}")
    else:
        n = build_lmdb(entries, images_dir, Path(args.out))
        print(f"  LMDB: {n} entries → {args.out}")


if __name__ == "__main__":
    main()
