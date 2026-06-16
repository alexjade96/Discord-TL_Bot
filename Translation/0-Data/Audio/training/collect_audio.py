"""Collect Discord audio submissions and transcription results for training.

Each successfully transcribed submission is appended to data/audio_submissions.jsonl.
Duplicates are detected by SHA-1 of the transcript text and silently skipped.

Schema:
    text_hash         -- SHA-1[:16] of transcript (deduplication key)
    filename          -- original Discord filename (e.g. voice-message.ogg)
    transcript        -- Whisper output text
    correct_transcript -- manual correction of transcript; used by future training
    translated_text   -- translation of transcript
    correct_translation -- manual correction of translation
    source_language   -- detected language code (e.g. 'ko', 'zh-cn')
    target_language   -- translation target (e.g. 'en')
    confidence        -- language detection confidence [0, 1] or None
    method            -- transcription method ('whisper-hf', 'whisper-local')
    username          -- Discord username of submitter
    timestamp         -- UTC ISO 8601
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_FILE = DATA_DIR / "audio_submissions.jsonl"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _existing_hashes() -> set[str]:
    if not SUBMISSIONS_FILE.exists():
        return set()
    hashes: set[str] = set()
    with SUBMISSIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                h = entry.get("text_hash")
                if h:
                    hashes.add(h)
            except (json.JSONDecodeError, Exception):
                pass
    return hashes


def save_submission(
    transcript: str,
    translated_text: str,
    source_language: str = "unknown",
    target_language: str = "en",
    confidence: float | None = None,
    method: str | None = None,
    username: str | None = None,
    filename: str = "",
) -> bool:
    """Save an audio transcription + translation submission to the dataset.

    Returns True if saved, False if skipped (empty transcript or duplicate).
    """
    if not transcript.strip():
        return False

    _ensure_dirs()

    text_hash = _text_hash(transcript)
    if text_hash in _existing_hashes():
        return False

    entry = {
        "text_hash": text_hash,
        "filename": filename,
        "transcript": transcript,
        "correct_transcript": None,     # manual ASR correction; used by future train.py
        "translated_text": translated_text,
        "correct_translation": None,    # manual translation correction
        "source_language": source_language,
        "target_language": target_language,
        "confidence": confidence,
        "method": method,
        "username": username,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with SUBMISSIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return True


def load_submissions() -> list[dict]:
    """Return all collected audio submissions as a list of dicts."""
    if not SUBMISSIONS_FILE.exists():
        return []
    with SUBMISSIONS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_stats() -> dict:
    """Return a summary of the current audio dataset."""
    submissions = load_submissions()
    languages: dict[str, int] = {}
    corrected_asr = 0
    corrected_tl = 0
    for entry in submissions:
        lang = entry.get("source_language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
        if entry.get("correct_transcript"):
            corrected_asr += 1
        if entry.get("correct_translation"):
            corrected_tl += 1
    return {
        "total": len(submissions),
        "corrected_asr": corrected_asr,
        "corrected_translation": corrected_tl,
        "languages": languages,
        "file": str(SUBMISSIONS_FILE),
    }


if __name__ == "__main__":
    stats = dataset_stats()
    print(f"Total submissions    : {stats['total']}")
    print(f"ASR corrections      : {stats['corrected_asr']}")
    print(f"Translation corrections: {stats['corrected_translation']}")
    print(f"File                 : {stats['file']}")
    print("Language breakdown:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
