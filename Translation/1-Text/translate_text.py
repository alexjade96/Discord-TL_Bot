"""Text translation via local fine-tuned models with HuggingFace Inference API fallback.

Local models are installed by 0-Data/Text/training/deploy.py into ~/.tl-bot/models/.
If a direction's local model is present it is used; otherwise the HF Inference API
is called instead. No code change is needed to activate a newly deployed model.

Usage (CLI):
    python translate_text.py "Меня зовут Вольфганг"
    python translate_text.py --src ru "Меня зовут Вольфганг"
    python translate_text.py --tgt fr "Hello world"

Environment:
    HF_TOKEN — HuggingFace Inference API token (required as fallback when no local model)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from huggingface_hub import InferenceClient

from detect import (
    detect_language_with_confidence,
    is_english,
    contains_non_latin,
    contains_cjk,
    segment_text,
)

# Multilingual → English (default fallback for languages without a dedicated model)
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-mul-en"
# English → target language (format with ISO 639-1 code, default pattern)
TRANSLATE_MODEL_TO = "Helsinki-NLP/opus-mt-en-{}"

# Language-specific model overrides: lang_code → (src_to_en_model, en_to_tgt_model)
# Used when the default mul-en or opus-mt-en-{tgt} pattern is unavailable or
# produces poor results for a specific language.
_LANG_MODEL_OVERRIDES: dict[str, tuple[str, str]] = {
    # opus-mt-mul-en produces "I'm sorry." for Korean — use dedicated ko→en model.
    # opus-mt-en-ko does not exist on HuggingFace — use Tatoeba Challenge big model.
    "ko": (
        "Helsinki-NLP/opus-mt-ko-en",
        "Helsinki-NLP/opus-mt-tc-big-en-ko",
    ),
}

# Local fine-tuned models installed by deploy.py
_LOCAL_MODELS_DIR = Path.home() / ".tl-bot" / "models"

# MarianMT tokenizer max sequence length; sequences longer than this are truncated.
_MAX_TOKEN_LENGTH = 512

# Module-level caches: direction → (model, tokenizer) to avoid reloading per call
_local_model_cache: dict[str, tuple] = {}


def _load_local_model(direction: str):
    """Return (MarianMTModel, MarianTokenizer) for a deployed direction, or None."""
    if direction in _local_model_cache:
        return _local_model_cache[direction]

    model_dir = _LOCAL_MODELS_DIR / direction
    if not model_dir.exists():
        _local_model_cache[direction] = None
        return None

    try:
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
        model = MarianMTModel.from_pretrained(str(model_dir))
        model.eval()
        _local_model_cache[direction] = (model, tokenizer)
        return _local_model_cache[direction]
    except Exception:
        _local_model_cache[direction] = None
        return None


def _run_local(text: str, model, tokenizer) -> str:
    """Run inference on a locally loaded MarianMT model."""
    import torch
    inputs = tokenizer([text], return_tensors="pt", padding=True,
                       truncation=True, max_length=_MAX_TOKEN_LENGTH)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="hf-inference", api_key=token)


def _translate_to_english(
    text: str,
    client: InferenceClient | None = None,
    src_lang: str | None = None,
) -> str:
    """Translate text to English.

    Resolution order:
    1. Local fine-tuned model for this direction (if deployed via deploy.py)
    2. Language-specific HF model from _LANG_MODEL_OVERRIDES (e.g. opus-mt-ko-en)
    3. Default multilingual model (opus-mt-mul-en) via HF API
    """
    lang_code = src_lang.split("-")[0] if src_lang else None
    override = _LANG_MODEL_OVERRIDES.get(lang_code) if lang_code else None

    # Direction key for local model: "ko-en" or "mul-en"
    local_direction = f"{lang_code}-en" if override else "mul-en"
    local = _load_local_model(local_direction)
    if local:
        return _run_local(text, *local)

    c = client or _client()
    if override:
        result = c.translation(text, model=override[0])
    elif src_lang:
        result = c.translation(text, model=TRANSLATE_MODEL,
                               src_lang=lang_code, tgt_lang="en")
    else:
        result = c.translation(text, model=TRANSLATE_MODEL)
    return result.translation_text


def _translate_from_english(
    text: str,
    tgt_lang: str,
    client: InferenceClient | None = None,
) -> str:
    """Translate English text to a target language.

    Resolution order:
    1. Local fine-tuned model for this direction (if deployed via deploy.py)
    2. Language-specific HF model from _LANG_MODEL_OVERRIDES (e.g. opus-mt-tc-big-en-ko)
    3. Default pattern model (opus-mt-en-{tgt}) via HF API

    tgt_lang is a langdetect-style code (e.g. 'fr', 'zh-cn', 'ja', 'ko').
    """
    tgt_code = tgt_lang.split("-")[0]
    override = _LANG_MODEL_OVERRIDES.get(tgt_code)

    local = _load_local_model(f"en-{tgt_code}")
    if local:
        return _run_local(text, *local)

    c = client or _client()
    hf_model = override[1] if override else TRANSLATE_MODEL_TO.format(tgt_code)
    result = c.translation(text, model=hf_model)
    return result.translation_text


# Sentence-boundary patterns: split *after* each terminator so punctuation
# stays attached to the sentence it ends (not the next one).
_SENT_CJK    = re.compile(r'(?<=[。？！…])')           # Chinese / Japanese
_SENT_LATIN  = re.compile(r'(?<=[.?!…])(?=[ \t])')    # Latin-script languages
_SENT_KOREAN = re.compile(                             # Korean: both terminator sets
    r'(?<=[。？！…\n])|(?<=[.?!])(?=[ \t])'
)


def _split_sentences(text: str, lang_code: str) -> list[str]:
    """Split text on script-appropriate sentence terminators.

    Returns a list of non-empty stripped chunks. Falls back to a single-element
    list containing the full text when no terminators are found.
    """
    base = lang_code.split("-")[0]
    if base in ("zh", "ja"):
        raw = _SENT_CJK.split(text)
    elif base == "ko":
        raw = _SENT_KOREAN.split(text)
    else:
        raw = _SENT_LATIN.split(text)
    chunks = [c.strip() for c in raw if c.strip()]
    return chunks or [text]


def _translate_chunked(text: str, lang_code: str, translate_fn) -> str:
    """Translate text sentence-by-sentence and rejoin.

    Splits on script-appropriate terminators, calls translate_fn on each chunk,
    then rejoins: empty string for CJK (no inter-sentence space), single space
    for all Latin-script languages.
    """
    chunks = _split_sentences(text, lang_code)
    if len(chunks) <= 1:
        return translate_fn(text)
    base = lang_code.split("-")[0]
    glue = "" if base in ("zh", "ja") else " "
    return glue.join(translate_fn(c) for c in chunks)


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
        # Override per-segment detection with the user's hint, but only when the
        # hint is a non-English language. When src_lang="en", CJK/script detection
        # (Unicode-range-based) is more reliable than the user's hint, and forcing
        # it to "en" would cause foreign segments to be passed through untranslated
        # before chaining to a non-English target language.
        if src_lang and src_lang != "en" and effective_lang != "en":
            effective_lang = src_lang

        if effective_lang == "en":
            parts.append(seg["text"])
        else:
            _el = effective_lang  # avoid late-binding closure capture
            parts.append(_translate_chunked(
                seg["text"], _el,
                lambda t, lang=_el: _translate_to_english(t, client, src_lang=lang),
            ))
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

    # Passthrough: auto-detected source matches the requested non-English target
    # (e.g. --to chinese on Chinese text). Base code comparison treats zh-cn/zh-tw as same.
    if not target_is_english and not source_is_english:
        if detected.split("-")[0] == tgt_lang.split("-")[0]:
            return {"translated_text": text, "source_language": detected, "confidence": confidence, "method": "passthrough"}

    client = _client()

    # English source -> non-English target: single hop, no segmentation needed
    if source_is_english and not target_is_english:
        translated = _translate_chunked(
            text, "en",
            lambda t: _translate_from_english(t, tgt_lang, client),
        )
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
        translated = _translate_chunked(
            text, detected,
            lambda t: _translate_to_english(t, client, src_lang=src_lang),
        )
        dominant_lang = detected
        method = "opus-mt"

    # Chain to non-English target if requested
    if not target_is_english:
        translated = _translate_chunked(
            translated, "en",
            lambda t: _translate_from_english(t, tgt_lang, client),
        )

    return {"translated_text": translated, "source_language": dominant_lang, "confidence": confidence, "method": method}


if __name__ == "__main__":
    import argparse
    import io as _io
    import sys as _sys
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Translate text.")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--src", default=None, help="Source language code (e.g. ru, zh-cn)")
    parser.add_argument("--tgt", default=None, help="Target language code (default: en)")
    args = parser.parse_args()

    result = translate_text(args.text, src_lang=args.src, tgt_lang=args.tgt)
    print(f"[{result['source_language']} -> {args.tgt or 'en'} via {result['method']}]")
    print(result["translated_text"])