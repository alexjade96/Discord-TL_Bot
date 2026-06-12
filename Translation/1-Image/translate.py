"""Full image translation pipeline: load image → OCR → detect language → translate.

Usage (CLI):
    python translate.py test_image_4.png
    python translate.py https://cdn.discordapp.com/.../image.png

Environment:
    HF_TOKEN — HuggingFace Inference API token (required for translation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Pull in the text translation pipeline from the sibling package
sys.path.insert(0, str(Path(__file__).parent.parent / "2-Text"))
from translate import translate_text  # noqa: E402

from ocr import extract_text_combined


def translate_image(
    source: str | Path | np.ndarray,
    src_lang: str | None = None,
) -> dict:
    """Extract text from an image and translate it to English.

    Args:
        source:   Discord attachment URL, local file path, or BGR numpy array.
        src_lang: Optional langdetect source language code override (e.g. 'ja').
                  Detected automatically when omitted.

    Returns:
        dict with keys:
            original_text   — raw OCR output (empty string if no text found)
            translated_text — English translation
            source_language — detected or provided langdetect code
            confidence      — language detection confidence [0, 1], or None
            ocr_confidence  — average EasyOCR confidence across all segments [0, 1]
            method          — 'none' | 'passthrough' | 'opus-mt'
    """
    original_text, ocr_confidence = extract_text_combined(source)

    if not original_text:
        return {
            "original_text": "",
            "translated_text": "",
            "source_language": "unknown",
            "confidence": None,
            "ocr_confidence": 0.0,
            "method": "none",
        }

    result = translate_text(original_text, src_lang=src_lang)

    return {
        "original_text": original_text,
        "translated_text": result["translated_text"],
        "source_language": result["source_language"],
        "confidence": result["confidence"],
        "ocr_confidence": ocr_confidence,
        "method": result["method"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text in an image to English.")
    parser.add_argument("source", help="Image file path or URL")
    parser.add_argument("--src", default=None, help="Source language code (e.g. ja, zh-cn)")
    args = parser.parse_args()

    result = translate_image(args.source, src_lang=args.src)
    print(f"OCR:         {result['original_text']}")
    print(f"OCR conf:    {result['ocr_confidence'] * 100:.1f}%")
    print(f"Language:    {result['source_language']} (detection: {result['confidence']})")
    print(f"Translation: {result['translated_text']}")
