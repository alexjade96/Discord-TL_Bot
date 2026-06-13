"""Image OCR using EasyOCR with OpenCV preprocessing.

Follows the approach from OCR_Models.ipynb:
  - Scale 2x with INTER_CUBIC before OCR for better accuracy on small text
  - Convert to grayscale
  - Three lazy EasyOCR readers (zh/ja/ko each paired with en); best confidence wins
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import requests
import easyocr
import wordninja

# Inserts a space at every letter↔digit boundary (e.g. "at522PM" → "at 522 PM").
# Numbers are script-neutral and splitting them out first lets wordninja focus
# on pure-letter runs without digit noise distorting its segmentation.
_DIGIT_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")

# Matches runs of 15+ consecutive Latin letters with no internal spaces.
# Shorter runs are left alone (likely a single word or intentional token).
_MERGED_WORD_RE = re.compile(r"[A-Za-z]{15,}")

# EasyOCR enforces that each CJK script can only share a reader with English.
# Three separate readers cover Chinese / Japanese / Korean; best result wins.
_reader_zh: easyocr.Reader | None = None  # ["en", "ch_sim"]
_reader_ja: easyocr.Reader | None = None  # ["en", "ja"]
_reader_ko: easyocr.Reader | None = None  # ["en", "ko"]


def _split_merged_words(text: str) -> str:
    """Split merged OCR tokens back into readable words.

    Two-pass approach:
    1. Split at every letter↔digit boundary so embedded numbers become
       separate tokens ("at522PM" → "at 522 PM"). Numbers are script-neutral
       and have no role in word segmentation, so isolating them first lets
       wordninja work on clean letter-only runs.
    2. Apply wordninja to any remaining letter run of 15+ chars to recover
       word boundaries lost by EasyOCR's line-merge ("thesoloq..." → "the solo q...").
    """
    text = _DIGIT_BOUNDARY_RE.sub(" ", text)
    return _MERGED_WORD_RE.sub(lambda m: " ".join(wordninja.split(m.group())), text)


def _get_reader_zh() -> easyocr.Reader:
    global _reader_zh
    if _reader_zh is None:
        _reader_zh = easyocr.Reader(["en", "ch_sim"], gpu=False)
    return _reader_zh


def _get_reader_ja() -> easyocr.Reader:
    global _reader_ja
    if _reader_ja is None:
        _reader_ja = easyocr.Reader(["en", "ja"], gpu=False)
    return _reader_ja


def _get_reader_ko() -> easyocr.Reader:
    global _reader_ko
    if _reader_ko is None:
        _reader_ko = easyocr.Reader(["en", "ko"], gpu=False)
    return _reader_ko


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


def preprocess_enhanced(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing: LANCZOS upscale → denoise → CLAHE → unsharp mask.

    Tuned conservatively vs the initial version — lower h, clipLimit, and sharpen
    strength avoid overcooking contrast on clean Discord screenshots where the
    baseline is already adequate.

    Adaptive thresholding (binary output) is intentionally omitted — EasyOCR's
    CRAFT+CRNN pipeline uses gradient magnitude in its feature maps, so converting
    to pure black/white removes information the model relies on.
    """
    scaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrast, 1.3, blurred, -0.3, 0)
    return sharpened


def _is_dark_mode(gray: np.ndarray) -> bool:
    """Return True when the image background is predominantly dark (Discord dark mode).

    Must be called on raw grayscale BEFORE CLAHE — CLAHE redistributes the
    histogram and makes brightness-based thresholds unreliable. Uses the
    75th-percentile rather than the mean so background pixels (which occupy
    more area than text) dominate the result.
    """
    return float(np.percentile(gray, 75)) < 128.0


def _remove_stripes(gray: np.ndarray) -> np.ndarray:
    """Erase thin horizontal separator lines between Discord chat sections.

    Uses morphological opening with a wide (1/2 image width), single-pixel-tall
    kernel. Only structures spanning at least half the image width survive erosion;
    individual character strokes are far shorter and are untouched. Detected
    separator pixels are painted white (background colour after normalisation).
    """
    w = gray.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 1), 1))
    lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = lines < 128
    cleaned = gray.copy()
    cleaned[mask] = 255
    return cleaned


def preprocess_discord(img: np.ndarray) -> np.ndarray:
    """Discord-aware preprocessing: tuned enhanced + dark-mode inversion + stripe removal.

    Discord-specific stages applied on raw grayscale before CLAHE:

    - Dark mode inversion: checked before CLAHE (CLAHE shifts the histogram and
      makes brightness thresholds unreliable post-application). When p75 < 128
      the image is inverted so EasyOCR sees the dark-text/light-background
      polarity it was trained on.
    - Stripe removal: only applied on light-mode images tall enough to contain
      multiple chat lines (scaled height >= 80px). In dark mode, inversion flips
      separator polarity from dark-on-dark to light-on-light, making the
      dark-structure detector unreliable. On very short images (single chat
      lines), the kernel would match actual character strokes spanning the width.
    """
    scaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    is_dark = _is_dark_mode(gray)
    if is_dark:
        gray = cv2.bitwise_not(gray)
    elif gray.shape[0] >= 80:
        gray = _remove_stripes(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrast, 1.3, blurred, -0.3, 0)
    return sharpened


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
    read_kwargs = dict(
        width_ths=1e4,      # merge wide text blocks into single lines
        add_margin=0.1,
        low_text=0.1,
        text_threshold=0.8,
        paragraph=False,
    )

    # Each CJK script requires its own reader. Run all three and pick the
    # one with the highest average confidence.
    candidates = [
        _get_reader_zh().readtext(processed, **read_kwargs),
        _get_reader_ja().readtext(processed, **read_kwargs),
        _get_reader_ko().readtext(processed, **read_kwargs),
    ]

    def _avg_conf(raw):
        confs = [c for _, text, c in raw if text.strip()]
        return sum(confs) / len(confs) if confs else 0.0

    results = max(candidates, key=_avg_conf)

    segments = []
    for bbox, text, confidence in results:
        text = _split_merged_words(text.strip())
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


# Maps langdetect codes to the reader getter that covers that script
_LANG_TO_READER = {
    "zh-cn": _get_reader_zh, "zh-tw": _get_reader_zh, "zh": _get_reader_zh,
    "ja": _get_reader_ja,
    "ko": _get_reader_ko,
}


def extract_text_hinted(
    processed: np.ndarray,
    lang_code: str,
    read_kwargs: dict,
) -> list[dict]:
    """Run OCR using the reader that best covers `lang_code`.

    Args:
        processed:   Preprocessed (grayscale, 2x) image array.
        lang_code:   langdetect-style code (e.g. 'zh-cn', 'ja', 'ko').
        read_kwargs: Shared readtext parameters.

    Returns:
        Parsed segment list (same format as extract_text).
    """
    getter = _LANG_TO_READER.get(lang_code)
    if getter is None:
        # Non-CJK: fall back to the reader with highest confidence across all three
        candidates = [
            _get_reader_zh().readtext(processed, **read_kwargs),
            _get_reader_ja().readtext(processed, **read_kwargs),
            _get_reader_ko().readtext(processed, **read_kwargs),
        ]

        def _avg(raw):
            confs = [c for _, t, c in raw if t.strip()]
            return sum(confs) / len(confs) if confs else 0.0

        raw = max(candidates, key=_avg)
    else:
        raw = getter().readtext(processed, **read_kwargs)

    segments = []
    for bbox, text, confidence in raw:
        text = _split_merged_words(text.strip())
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
