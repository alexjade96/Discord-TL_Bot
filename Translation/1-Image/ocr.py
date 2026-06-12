"""Image OCR using EasyOCR with OpenCV preprocessing.

Follows the approach from OCR_Models.ipynb:
  - Scale 2x with INTER_CUBIC before OCR for better accuracy on small text
  - Convert to grayscale
  - EasyOCR with English + CJK language set (lazy-initialised)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import requests
import easyocr

# Languages loaded into EasyOCR reader — extend as needed
OCR_LANGUAGES = ["en", "ch_sim", "ja", "ko"]

# Lazy-initialised; loading EasyOCR models takes several seconds
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _reader


def load_image_from_url(url: str, timeout: int = 15) -> np.ndarray:
    """Download an image from a URL and return as an OpenCV BGR array."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    arr = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image downloaded from: {url}")
    return img


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Load an image from a file path and return as an OpenCV BGR array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Scale 2x (INTER_CUBIC) and convert to grayscale for better OCR accuracy."""
    scaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)


def extract_text(source: str | Path | np.ndarray) -> list[dict]:
    """Run EasyOCR on an image and return detected text segments.

    Args:
        source: Discord attachment URL, local file path, or pre-loaded BGR numpy array.

    Returns:
        List of dicts sorted in top-to-bottom reading order, each with:
            text        — detected string
            confidence  — OCR confidence [0, 1]
            bbox        — four corner points [[x,y], ...]
    """
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        img = load_image_from_url(source)
    else:
        img = load_image_from_path(source)

    processed = preprocess(img)
    results = _get_reader().readtext(
        processed,
        width_ths=1e4,      # merge wide text blocks into single lines
        add_margin=0.1,
        low_text=0.1,
        text_threshold=0.8,
        paragraph=False,
    )

    segments = []
    for bbox, text, confidence in results:
        text = text.strip()
        if not text:
            continue
        segments.append({
            "text": text,
            "confidence": round(confidence, 4),
            "bbox": bbox,
            "_top_y": min(pt[1] for pt in bbox),
        })

    segments.sort(key=lambda s: s["_top_y"])
    for s in segments:
        del s["_top_y"]

    return segments


def extract_text_combined(source: str | Path | np.ndarray) -> tuple[str, float]:
    """Extract all text from an image as a single string with average OCR confidence.

    Returns:
        (combined_text, avg_confidence) — avg_confidence is 0.0 if no text found.
    """
    segments = extract_text(source)
    if not segments:
        return "", 0.0
    combined = " ".join(s["text"] for s in segments)
    avg_conf = sum(s["confidence"] for s in segments) / len(segments)
    return combined, round(avg_conf, 4)
