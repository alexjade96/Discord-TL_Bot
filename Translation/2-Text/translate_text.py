"""Text translation via HuggingFace Inference API.

Usage (CLI):
    python translate_text.py "Меня зовут Вольфганг"
    python translate_text.py --src ru "Меня зовут Вольфганг"
    python translate_text.py --tgt fr "Hello world"

Environment:
    HF_TOKEN — HuggingFace Inference API token (required)
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

from detect import (
    detect_language_with_confidence,
    is_english,
    contains_non_latin,
    contains_cjk,
    segment_text,
)

# Multilingual → English
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-mul-en"
# English → target language (format with ISO 639-1 code)
TRANSLATE_MODEL_TO = "Helsinki-NLP/opus-mt-en-{}"


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="hf-inference", api_key=token)


def translate_to_english(
    text: str,
    client: InferenceClient | None = None,
    src_lang: str | None = None,
) -> str:
    """Translate text to English using Helsinki-NLP/opus-mt-mul-en."""
    c = client or _client()
    if src_lang:
        api_lang = src_lang.split("-")[0]
        result = c.translation(text, model=TRANSLATE_MODEL, src_lang=api_lang, tgt_lang="en")
    else:
        result = c.translation(text, model=TRANSLATE_MODEL)
    return result.translation_text


def translate_from_english(
    text: str,
    tgt_lang: str,
    client: InferenceClient | None = None,
) -> str:
    """Translate English text to a target language using Helsinki-NLP/opus-mt-en-{tgt}.

    tgt_lang is a langdetect-style code (e.g. 'fr', 'zh-cn', 'ja').
    Uses the bare ISO 639-1 code as the model suffix.
    """
    c = client or _client()
    tgt_code = tgt_lang.split("-")[0]
    model = TRANSLATE_MODEL_TO.format(tgt_code)
    result = c.translation(text, model=model)
    return result.translation_text


def _translate_via_segments(
    segs: list[dict],
    client: InferenceClient,
    src_lang: str | None = None,
) -> tuple[str, str]:
    """Translate each non-English segment in place, leaving English spans unchanged.

    Works on a pre-computed segment list from segment_text so segmentation runs
    once per translate_text call. Handles all mixed-script cases (CJK, Cyrillic,
    Arabic, etc.) through the same code path.

    Returns (translated_text, dominant_lang_code) where dominant_lang is the
    lang_code of the first foreign segment encountered.
    """
    parts = []
    dominant_lang = src_lang or "unknown"
    found_foreign = False

    for seg in segs:
        effective_lang = seg["lang_code"]
        if src_lang and effective_lang != "en":
            effective_lang = src_lang

        if effective_lang == "en":
            parts.append(seg["text"])
        else:
            parts.append(translate_to_english(seg["text"], client, src_lang=effective_lang))
            if not found_foreign:
                dominant_lang = seg["lang_code"]
                found_foreign = True

    return "".join(parts), dominant_lang


def translate_text(
    text: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
) -> dict:
    """Detect language and translate text.

    Translates to English by default. If tgt_lang is provided and is not English,
    chains a second pass: source -> English -> target language.

    Mixed-script text (CJK+Latin, Arabic+Latin, Cyrillic+Latin, etc.) is segmented
    by script and language via segment_text, with each non-English span translated
    individually and Latin content left in place.

    Args:
        text:     Input text to translate.
        src_lang: Optional langdetect-style source language code (e.g. 'ru', 'zh-cn').
                  Detected automatically when omitted.
        tgt_lang: Optional langdetect-style target language code. Defaults to English.

    Returns:
        dict with keys:
            translated_text  -- output in target language
            source_language  -- detected or provided langdetect code
            confidence       -- detection confidence [0, 1], or None if src_lang provided
            method           -- 'none' | 'passthrough' | 'opus-mt' | 'opus-mt-segmented'
    """
    if not text.strip():
        return {"translated_text": text, "source_language": "unknown", "confidence": None, "method": "none"}

    if src_lang:
        detected, confidence = src_lang, None
    else:
        detected, confidence = detect_language_with_confidence(text)

    has_mixed = contains_non_latin(text) or contains_cjk(text)
    target_is_english = not tgt_lang or tgt_lang == "en"
    source_is_english = (
        (detected == "en" or (not src_lang and is_english(text)))
        and not has_mixed
    )

    # Passthrough: source and target are both English
    if source_is_english and target_is_english:
        return {"translated_text": text, "source_language": "en", "confidence": confidence, "method": "passthrough"}

    client = _client()

    # English source -> non-English target: single hop, no segmentation needed
    if source_is_english and not target_is_english:
        translated = translate_from_english(text, tgt_lang, client)
        return {"translated_text": translated, "source_language": "en", "confidence": confidence, "method": "opus-mt"}

    # Mixed or pure foreign text: segment once, translate non-English spans in place
    segs = segment_text(text)
    foreign_segs = [s for s in segs if s["lang_code"] != "en"]
    english_segs = [s for s in segs if s["lang_code"] == "en"]

    if foreign_segs:
        translated, dominant_lang = _translate_via_segments(segs, client, src_lang)
        method = "opus-mt-segmented" if english_segs else "opus-mt"
    else:
        # segment_text sees all English but detection disagrees (short/ambiguous text)
        translated = translate_to_english(text, client, src_lang=src_lang)
        dominant_lang = detected
        method = "opus-mt"

    # Chain to non-English target if requested
    if not target_is_english:
        translated = translate_from_english(translated, tgt_lang, client)

    return {"translated_text": translated, "source_language": dominant_lang, "confidence": confidence, "method": method}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text.")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--src", default=None, help="Source language code (e.g. ru, zh-cn)")
    parser.add_argument("--tgt", default=None, help="Target language code (default: en)")
    args = parser.parse_args()

    result = translate_text(args.text, src_lang=args.src, tgt_lang=args.tgt)
    print(f"[{result['source_language']} -> {args.tgt or 'en'} via {result['method']}]")
    print(result["translated_text"])