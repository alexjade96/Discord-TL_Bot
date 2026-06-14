"""Collect Discord text submissions and translation results for training.

Each successfully translated submission is appended to data/text_submissions.jsonl.
Duplicates are detected by SHA-1 of the source text and silently skipped.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_FILE = DATA_DIR / "text_submissions.jsonl"


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
    original_text: str,
    translated_text: str,
    source_language: str = "unknown",
    target_language: str = "en",
    confidence: float | None = None,
    method: str | None = None,
    username: str | None = None,
) -> bool:
    """Save a text translation submission to the training dataset.

    Args:
        original_text:   Source text as submitted.
        translated_text: Bot's translation output.
        source_language: Detected language code (e.g. 'zh-cn', 'ja').
        target_language: Target language code (e.g. 'en', 'fr').
        confidence:      Language detection confidence [0, 1], or None.
        method:          Translation method ('opus-mt', 'opus-mt-segmented', etc.).
        username:        Discord username of the submitter.

    Returns:
        True if saved, False if skipped (empty text or duplicate).
    """
    if not original_text.strip():
        return False

    _ensure_dirs()

    text_hash = _text_hash(original_text)
    if text_hash in _existing_hashes():
        return False

    entry = {
        "text_hash": text_hash,
        "original_text": original_text,
        "translated_text": translated_text,
        "correct_translation": None, # manual correction of translated_text; used in future training
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
    """Return all collected text submissions as a list of dicts."""
    if not SUBMISSIONS_FILE.exists():
        return []
    with SUBMISSIONS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_stats() -> dict:
    """Return a summary of the current text dataset."""
    submissions = load_submissions()
    languages: dict[str, int] = {}
    for entry in submissions:
        lang = entry.get("source_language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
    return {
        "total": len(submissions),
        "languages": languages,
        "file": str(SUBMISSIONS_FILE),
    }


if __name__ == "__main__":
    stats = dataset_stats()
    print(f"Total submissions: {stats['total']}")
    print(f"File:              {stats['file']}")
    print("Language breakdown:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
