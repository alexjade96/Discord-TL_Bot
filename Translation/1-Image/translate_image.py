"""Full image translation pipeline: load image -> OCR -> detect language -> translate.

When a language hint is supplied, the pipeline runs two passes:
  - Auto pass:  best-confidence reader selected automatically
  - Hint pass:  reader targeting the hinted language script
Both are translated and scored by ocr_confidence * lang_confidence.
The higher-scoring result is returned, with hint_used flagged in the dict.

Usage (CLI):
    python translate_image.py test_image_4.png
    python translate_image.py image.png --src ja
    python translate_image.py image.png --hint chinese

Environment:
    HF_TOKEN -- HuggingFace Inference API token (required for translation)
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)
from pathlib import Path

import numpy as np

# Pull in the text translation pipeline from the sibling package
sys.path.insert(0, str(Path(__file__).parent.parent / "2-Text"))
from translate_text import translate_text  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "2-Text"))
from detect import confidence_for_language  # noqa: E402

from ocr import (
    extract_text_combined,
    extract_text_hinted,
    load_image_from_url,
    load_image_from_path,
    preprocess,
)
sys.path.insert(0, str(Path(__file__).parent.parent / "0-Data" / "Image" / "training"))
from collect_image import save_submission  # noqa: E402

_READ_KWARGS = dict(
    width_ths=1e4,
    add_margin=0.1,
    low_text=0.1,
    text_threshold=0.8,
    paragraph=False,
)


def _combined_score(ocr_conf: float, lang_conf: float) -> float:
    """Score a pass as ocr_confidence * language_confidence."""
    return ocr_conf * lang_conf


def _run_pass(
    raw_image: np.ndarray,
    processed: np.ndarray,
    src_lang: str | None,
    hint_lang: str | None,
    tgt_lang: str | None = None,
) -> dict:
    """Run OCR (with optional hint-specific reader) then translate.

    Returns the full result dict plus ocr_confidence and score.
    """
    if hint_lang:
        segments = extract_text_hinted(processed, hint_lang, _READ_KWARGS)
        if segments:
            ocr_text = " ".join(s["text"] for s in segments)
            ocr_conf = round(sum(s["confidence"] for s in segments) / len(segments), 4)
        else:
            ocr_text, ocr_conf = "", 0.0
    else:
        ocr_text, ocr_conf = extract_text_combined(raw_image)

    if not ocr_text:
        return {
            "original_text": "",
            "translated_text": "",
            "source_language": "unknown",
            "confidence": None,
            "ocr_confidence": 0.0,
            "method": "none",
            "_score": 0.0,
        }

    result = translate_text(ocr_text, src_lang=src_lang, tgt_lang=tgt_lang)
    lang_conf = result.get("confidence")

    if hint_lang is not None:
        # Replace the None confidence (no detection was run) with langdetect's
        # actual probability for the hinted language in the detected candidates.
        # Falls back to 0.0 if langdetect didn't consider it at all.
        lang_conf = confidence_for_language(ocr_text, hint_lang)

    score = _combined_score(ocr_conf, lang_conf or 0.0)

    return {
        "original_text": ocr_text,
        "translated_text": result["translated_text"],
        "source_language": result["source_language"],
        "confidence": lang_conf,
        "ocr_confidence": ocr_conf,
        "method": result["method"],
        "_score": score,
    }


def translate_image(
    source: str | Path | np.ndarray,
    src_lang: str | None = None,
    hint_lang: str | None = None,
    tgt_lang: str | None = None,
    original_filename: str | None = None,
    username: str | None = None,
) -> dict:
    """Extract text from an image and translate it to English.

    When hint_lang is provided, both an auto-detect pass and a hint-targeted
    pass are run. Both results are returned together so the caller can present
    them side-by-side for comparison.

    Args:
        source:            Discord attachment URL, local file path, or BGR numpy array.
        src_lang:          Hard override for the source language code (skips detection).
        hint_lang:         User-suggested langdetect code (e.g. 'zh-cn', 'ja', 'ko').
                           Triggers a second hinted pass alongside the auto pass.
        original_filename: Discord attachment filename; used to name the saved image.
        username:          Discord username of the submitter; included in saved filename.

    Returns:
        dict with keys:
            auto  -- result dict from the auto-detect pass (always present)
            hint  -- result dict from the hinted pass, or None if no hint given

        Each result dict contains:
            original_text   -- raw OCR output
            translated_text -- English translation
            source_language -- detected or provided langdetect code
            confidence      -- language detection confidence [0, 1], or None
            ocr_confidence  -- average EasyOCR confidence [0, 1]
            method          -- 'none' | 'passthrough' | 'opus-mt' | 'opus-mt-segmented'
            score           -- combined confidence (ocr_conf * lang_conf)
    """
    if isinstance(source, np.ndarray):
        raw_image = source
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        raw_image = load_image_from_url(source)
    else:
        raw_image = load_image_from_path(source)

    processed = preprocess(raw_image)

    auto = _run_pass(raw_image, processed, src_lang=src_lang, hint_lang=None, tgt_lang=tgt_lang)
    auto["score"] = auto.pop("_score")

    hint = None
    if hint_lang and hint_lang != src_lang:
        hint = _run_pass(raw_image, processed, src_lang=hint_lang, hint_lang=hint_lang, tgt_lang=tgt_lang)
        hint["score"] = hint.pop("_score")
    else:
        auto.pop("_score", None)

    # Save the auto pass to the training dataset (non-fatal)
    collected_path: str | None = None
    try:
        saved = save_submission(
            raw_image,
            ocr_text=auto["original_text"],
            source_language=auto["source_language"],
            confidence=auto["confidence"],
            ocr_confidence=auto["ocr_confidence"],
            original_filename=original_filename,
            username=username,
        )
        if saved is not None:
            collected_path = str(saved)
    except Exception:
        logger.warning("Failed to save image to training dataset", exc_info=True)

    return {"auto": auto, "hint": hint, "collected_path": collected_path}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text in an image to English.")
    parser.add_argument("source", help="Image file path or URL")
    parser.add_argument("--src", default=None, help="Hard source language override (e.g. ja)")
    parser.add_argument("--hint", default=None, help="Language hint code (triggers dual-pass)")
    args = parser.parse_args()

    result = translate_image(args.source, src_lang=args.src, hint_lang=args.hint)
    auto = result["auto"]
    hint = result["hint"]
    print(f"OCR:         {auto['original_text']}")
    print(f"OCR conf:    {auto['ocr_confidence'] * 100:.1f}%")
    print(f"Language:    {auto['source_language']} (conf: {auto['confidence']})")
    print(f"Translation: {auto['translated_text']}")
    print(f"Method:      {auto['method']}  |  score: {auto.get('score', 0):.2f}")
    if hint and hint["method"] != "none":
        print(f"\nHint pass:")
        print(f"  OCR:         {hint['original_text']}")
        print(f"  Translation: {hint['translated_text']}")
        print(f"  Method:      {hint['method']}  |  score: {hint.get('score', 0):.2f}")
