from __future__ import annotations

from langdetect import detect, detect_langs, LangDetectException, DetectorFactory
from lingua import Language, LanguageDetectorBuilder

DetectorFactory.seed = 0  # make langdetect deterministic across runs

from utils import get_mbart_code

# Lazy-initialised; building the detector is expensive (~1s)
_detector: LanguageDetectorBuilder | None = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_all_languages().build()
    return _detector


def detect_language(text: str) -> str:
    """Return a langdetect-style language code (e.g. 'zh-cn', 'ru', 'en').

    Falls back to 'unknown' when detection fails (too short / ambiguous).
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_language_segments(text: str) -> list[dict]:
    """Detect languages per segment for mixed-language text using lingua.

    Returns a list of dicts with keys: language, confidence, text, start, end.
    Segments map directly to lingua's detected spans.
    """
    detector = _get_detector()
    results = detector.detect_multiple_languages_of(text)
    segments = []
    for r in results:
        segment_text = text[r.start_index : r.end_index]
        lang = r.language
        confidence = detector.compute_language_confidence(segment_text, lang)
        # Small boost for English — langdetect is biased against short English words
        if lang == Language.ENGLISH:
            confidence = min(confidence * 1.1, 1.0)
        segments.append(
            {
                "language": lang.name.lower(),
                "mbart_code": get_mbart_code(lang.iso_code_639_1.name.lower()),
                "confidence": round(confidence, 4),
                "text": segment_text,
                "start": r.start_index,
                "end": r.end_index,
            }
        )
    return segments


def is_english(text: str) -> bool:
    """Return True if the entire text is detected as English."""
    return detect_language(text) == "en"


def dominant_language(text: str) -> str:
    """Return the langdetect code of the highest-probability language."""
    try:
        langs = detect_langs(text)
        return langs[0].lang if langs else "unknown"
    except LangDetectException:
        return "unknown"
