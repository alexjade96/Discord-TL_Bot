"""Collect Discord-submitted images and OCR labels for training.

Each successfully OCR'd image is saved to data/ and its metadata
appended to data/labels.jsonl. This dataset feeds train.py.

Duplicates are detected by SHA-1 of raw image bytes and silently skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
LABELS_FILE = DATA_DIR / "labels.jsonl"

# Characters not safe for filenames — replaced with underscores
_UNSAFE_RE = re.compile(r"[^\w\-]")


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _image_hash(img: np.ndarray) -> str:
    return hashlib.sha1(img.tobytes()).hexdigest()[:16]


def _existing_hashes() -> set[str]:
    """Return the set of image_hash values already recorded in labels.jsonl.

    Falls back to extracting the hash from legacy filenames (last underscore
    segment) so old entries still participate in duplicate detection.
    """
    if not LABELS_FILE.exists():
        return set()
    hashes: set[str] = set()
    with LABELS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                h = entry.get("image_hash")
                if h:
                    hashes.add(h)
                else:
                    # Legacy filenames: YYYYMMDD_HHMMSS_<hash16>.png
                    stem = Path(entry.get("filename", "")).stem
                    hashes.add(stem.split("_")[-1])
            except (json.JSONDecodeError, Exception):
                pass
    return hashes


def _build_filename(
    img_hash: str,
    original_filename: str | None,
    username: str | None,
) -> str:
    """Construct a filename: YYYYMMDD[_username][_original_stem].png"""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    parts = [date]
    if username:
        parts.append(_UNSAFE_RE.sub("_", username)[:24])
    if original_filename:
        stem = _UNSAFE_RE.sub("_", Path(original_filename).stem)[:48]
        parts.append(stem)
    else:
        parts.append(img_hash[:8])
    return "_".join(parts) + ".png"


def save_submission(
    image: np.ndarray,
    ocr_text: str,
    source_language: str = "unknown",
    confidence: float | None = None,
    ocr_confidence: float | None = None,
    original_filename: str | None = None,
    username: str | None = None,
) -> Path | None:
    """Save a raw image and its OCR label to the training dataset.

    Args:
        image:             Raw BGR numpy array (before preprocessing).
        ocr_text:          OCR-extracted text string.
        source_language:   Detected language code (e.g. 'ko', 'zh-cn').
        confidence:        Language detection confidence [0, 1], or None.
        ocr_confidence:    Average OCR confidence across segments [0, 1], or None.
        original_filename: Discord attachment filename (e.g. 'screenshot.png').
        username:          Discord username of the submitter.

    Returns:
        Path to the saved image, or None if skipped (empty text or duplicate).
    """
    if not ocr_text.strip():
        return None

    _ensure_dirs()

    img_hash = _image_hash(image)
    if img_hash in _existing_hashes():
        return None

    filename = _build_filename(img_hash, original_filename, username)
    img_path = DATA_DIR / filename

    if not cv2.imwrite(str(img_path), image):
        raise OSError(f"cv2.imwrite failed for path: {img_path}")

    entry = {
        "filename": filename,
        "image_hash": img_hash,
        "ocr_text": ocr_text,
        "source_language": source_language,
        "confidence": confidence,
        "ocr_confidence": ocr_confidence,
        "username": username,
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
        "images_dir": str(DATA_DIR),
    }


if __name__ == "__main__":
    stats = dataset_stats()
    print(f"Total submissions: {stats['total']}")
    print(f"Images directory:  {stats['images_dir']}")
    print("Language breakdown:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
