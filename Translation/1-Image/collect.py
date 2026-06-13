"""Collect Discord-submitted images and OCR labels for training.

Each successfully OCR'd image is saved to data/images/ and its metadata
appended to data/labels.jsonl. This dataset feeds training/train.py.

Duplicates are detected by SHA-1 of raw image bytes and silently skipped.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.jsonl"


def _ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _image_hash(img: np.ndarray) -> str:
    return hashlib.sha1(img.tobytes()).hexdigest()[:16]


def save_submission(
    image: np.ndarray,
    ocr_text: str,
    source_language: str = "unknown",
    confidence: float | None = None,
    ocr_confidence: float | None = None,
) -> Path | None:
    """Save a raw image and its OCR label to the training dataset.

    Args:
        image:           Raw BGR numpy array (before preprocessing).
        ocr_text:        OCR-extracted text string.
        source_language: Detected language code (e.g. 'ko', 'zh-cn').
        confidence:      Language detection confidence [0, 1], or None.
        ocr_confidence:  Average OCR confidence across segments [0, 1], or None.

    Returns:
        Path to the saved image, or None if skipped (empty text or duplicate).
    """
    if not ocr_text.strip():
        return None

    _ensure_dirs()

    img_hash = _image_hash(image)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{img_hash}.png"
    img_path = IMAGES_DIR / filename

    # Skip duplicates — same pixel content submitted again
    existing = {p.stem.split("_")[-1] for p in IMAGES_DIR.glob("*.png")}
    if img_hash in existing:
        return None

    cv2.imwrite(str(img_path), image)

    entry = {
        "filename": filename,
        "ocr_text": ocr_text,
        "source_language": source_language,
        "confidence": confidence,
        "ocr_confidence": ocr_confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with LABELS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return img_path


def load_labels() -> list[dict]:
    """Return all collected labels as a list of dicts."""
    if not LABELS_FILE.exists():
        return []
    with LABELS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_stats() -> dict:
    """Return a summary of the current dataset."""
    labels = load_labels()
    languages: dict[str, int] = {}
    for entry in labels:
        lang = entry.get("source_language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
    return {
        "total": len(labels),
        "languages": languages,
        "images_dir": str(IMAGES_DIR),
    }


if __name__ == "__main__":
    stats = dataset_stats()
    print(f"Total submissions: {stats['total']}")
    print(f"Images directory:  {stats['images_dir']}")
    print("Language breakdown:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
