from __future__ import annotations

import re
import unicodedata

from langdetect import detect, detect_langs, LangDetectException, DetectorFactory
from lingua import Language, LanguageDetectorBuilder

DetectorFactory.seed = 0  # make langdetect deterministic across runs

from utils import get_mbart_code, get_language_name

# Lazy-initialised; building the detector is expensive (~1s)
_detector = None

# Strips basic Latin letters (A-Z, a-z, Latin Extended U+00C0-U+024F) from text.
# Whatever remains after stripping is non-Latin script (CJK, Hangul, Arabic, etc.).
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+")

# Matches CJK Unified Ideographs (U+3000-U+9FFF), Hangul (U+AC00-U+D7A3),
# CJK Compatibility Ideographs (U+F900-U+FAFF).
_CJK_SEQ_RE = re.compile("[　-鿿가-힣豈-﫿]+")

# Tokens that carry no language signal: @mentions and digit sequences.
# Stripped before running lingua so they don't skew n-gram detection.
_NOISE_TOKEN_RE = re.compile(r"@\S+|\d+")

# Script-specific ranges that unambiguously identify Korean and Japanese.
# CJK Unified Ideographs alone are shared across zh/ja/ko so they need langdetect.
_HANGUL_RE = re.compile("[가-힣ᄀ-ᇿ㄰-㆏]")
_KANA_RE = re.compile("[぀-ヿ]")


def _get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_all_languages().build()
    return _detector


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_language_segments(text):
    """Detect languages per segment for mixed-language text using lingua.

    Returns a list of dicts with keys: language, confidence, text, start, end.
    Segments map directly to lingua's detected spans.
    """
    detector = _get_detector()
    results = detector.detect_multiple_languages_of(text)
    segments = []
    for r in results:
        span = text[r.start_index : r.end_index]
        lang = r.language
        confidence = detector.compute_language_confidence(span, lang)
        segments.append(
            {
                "language": lang.name.lower(),
                "mbart_code": get_mbart_code(lang.iso_code_639_1.name.lower()),
                "confidence": round(confidence, 4),
                "text": span,
                "start": r.start_index,
                "end": r.end_index,
            }
        )
    return segments


def contains_non_latin(text, threshold=0.1):
    """Return True if more than `threshold` fraction of letters are non-Latin."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_latin = [c for c in letters if unicodedata.category(c) == "Lo" or ord(c) > 0x024F]
    return len(non_latin) / len(letters) > threshold


def contains_cjk(text: str) -> bool:
    """Return True if the text contains any CJK character sequence."""
    return bool(_CJK_SEQ_RE.search(text))


def is_english(text):
    """Return True only if the text is purely/dominantly English with no non-Latin script."""
    if contains_non_latin(text):
        return False
    result = _get_detector().detect_language_of(text)
    return result == Language.ENGLISH


def dominant_language(text):
    """Return the langdetect code of the highest-probability language."""
    try:
        langs = detect_langs(text)
        return langs[0].lang if langs else "unknown"
    except LangDetectException:
        return "unknown"


def strip_english_segments(text):
    """Remove Latin/English content from mixed-language text before langdetect.

    For CJK-containing text, extracts CJK character sequences directly using
    Unicode ranges. For non-CJK non-Latin text, falls back to lingua segmentation.
    Falls back to the original text when nothing can be stripped.
    """
    if not contains_non_latin(text):
        return text

    cjk_parts = _CJK_SEQ_RE.findall(text)
    if cjk_parts:
        return " ".join(cjk_parts)

    segments = detect_language_segments(text)
    if not segments:
        return text

    foreign_parts = [s["text"] for s in segments if s["language"] != "english"]
    return " ".join(foreign_parts) if foreign_parts else text


def confidence_for_language(text, lang_code):
    """Return langdetect's probability for a specific language code.

    Strips Latin segments first so mixed-script text does not pollute the result.
    Returns 0.0 if langdetect does not consider that language at all.
    """
    filtered = strip_english_segments(text)
    try:
        for lang in detect_langs(filtered):
            if lang.lang == lang_code:
                return round(lang.prob, 4)
        return 0.0
    except LangDetectException:
        return 0.0


def detect_language_with_confidence(text):
    """Return (langdetect_code, confidence) for the most probable language.

    Strips Latin segments first on mixed-script input so langdetect sees only
    the foreign-language content.
    """
    filtered = strip_english_segments(text)
    try:
        langs = detect_langs(filtered)
        if langs:
            return langs[0].lang, round(langs[0].prob, 4)
        return "unknown", 0.0
    except LangDetectException:
        return "unknown", 0.0


def _detect_cjk_language(text: str) -> str:
    """Return a langdetect code for a CJK text span.

    Script-specific Unicode ranges are used as the primary signal — they are
    unambiguous and reliable even for single characters. Langdetect is only
    consulted to separate Chinese from Japanese when no script-unique characters
    are present. A Korean result from langdetect is vetoed when no Hangul exists
    in the text, defaulting to Chinese (the dominant CJK Unified Ideograph user).
    """
    if _HANGUL_RE.search(text):
        return "ko"
    if _KANA_RE.search(text):
        return "ja"
    # Pure CJK Unified Ideographs: use langdetect to distinguish zh vs ja
    code, _ = detect_language_with_confidence(text)
    if code == "ko":
        return "zh-cn"
    return code


def _classify_latin_segment(text: str) -> dict:
    """Classify a single Latin-script span, returning a segment dict.

    Strips @mentions and digit sequences before detection — they carry no language
    signal and skew lingua's n-gram model toward spurious matches. Defaults to
    English when lingua cannot make a determination.
    """
    cleaned = _NOISE_TOKEN_RE.sub("", text).strip()
    if not cleaned:
        return {"text": text, "lang_code": "en", "lang_name": "English"}
    try:
        lang = _get_detector().detect_language_of(cleaned)
        if lang is not None:
            code = lang.iso_code_639_1.name.lower()
            return {"text": text, "lang_code": code, "lang_name": get_language_name(code)}
    except Exception:
        pass
    return {"text": text, "lang_code": "en", "lang_name": "English"}


def _segment_latin(text: str) -> list[dict]:
    """Classify a Latin-script gap as a single segment.

    Discord chat gaps between CJK spans are typically short and contain noise
    (timestamps, @mentions, numbers) that causes lingua's multi-language boundary
    detection to misfire. Treating each gap as a single unit and classifying it
    via _classify_latin_segment (which strips noise before detection) is more
    reliable for this input type.
    """
    return [_classify_latin_segment(text)]


def segment_text(text: str) -> list[dict]:
    """Segment text by script and language, returning an ordered list of spans.

    CJK character sequences are identified by Unicode range (reliable at any
    proportion). Latin-script spans between CJK runs are split by lingua's
    multi-language detection where possible, then classified individually.

    This is the single segmentation function used by both the translation pipeline
    and the analysis/display path.

    Returns:
        Ordered list of dicts, each with:
            text      — the span text (preserves original spacing/content)
            lang_code — langdetect-style code (e.g. 'en', 'zh-cn', 'ja', 'ko', 'ru')
            lang_name — human-readable name (e.g. 'English', 'Chinese (Simplified)')
    """
    result = []
    last_end = 0

    for match in _CJK_SEQ_RE.finditer(text):
        gap = text[last_end : match.start()]
        if gap.strip():
            result.extend(_segment_latin(gap))

        cjk = match.group()
        code = _detect_cjk_language(cjk)
        result.append({"text": cjk, "lang_code": code, "lang_name": get_language_name(code)})
        last_end = match.end()

    tail = text[last_end:]
    if tail.strip():
        result.extend(_segment_latin(tail))

    return result


def analyze_segments(text: str) -> dict[str, list[str]]:
    """Segment text by script and language, returning spans grouped by language name.

    Thin wrapper over segment_text that groups the ordered segments by language
    for display purposes.

    Returns:
        dict mapping language name -> list of text segments in that language,
        in the order they first appear. E.g.:
            {
                "English": ["Jon Yesterday at 5:22PM @coach hug", "the solo q streamer..."],
                "Chinese (Simplified)": ["赢家的视角"],
            }
    """
    grouped: dict[str, list[str]] = {}
    for seg in segment_text(text):
        s = seg["text"].strip()
        if s:
            grouped.setdefault(seg["lang_name"], []).append(s)
    return grouped