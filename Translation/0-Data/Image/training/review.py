"""Interactive CLI for manually reviewing and correcting collected training entries.

Walks through uncorrected entries in labels.jsonl (image mode) or
text_submissions.jsonl (text mode), prompting for corrections. Rewrites
the JSONL in-place with updated fields so dataset.py picks them up on
the next build.

Whether a correction was made is inferred from the data: a non-null
correct_text / correct_translation field means a human has overridden
the bot's output. No separate "reviewed" flag is needed.

Usage:
    python review.py                        # review image OCR entries
    python review.py --mode text            # review text translation entries
    python review.py --all                  # re-review already-corrected entries
    python review.py --stats                # print correction coverage and exit
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 so CJK characters display correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
IMAGE_DATA_DIR = HERE.parent / "data"
TEXT_DATA_DIR = HERE.parent.parent.parent / "0-Data" / "Text" / "data"

IMAGE_LABELS = IMAGE_DATA_DIR / "labels.jsonl"
TEXT_LABELS = TEXT_DATA_DIR / "text_submissions.jsonl"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, current: str | None = None) -> str | None:
    """Show current value, return new value or None to keep unchanged."""
    if current:
        print(f"  Current : {current}")
    raw = input(f"  {label} (Enter to keep, text to replace, '-' to clear): ").strip()
    if raw == "-":
        return ""
    return raw if raw else None


# ---------------------------------------------------------------------------
# Image review
# ---------------------------------------------------------------------------

def _print_image_entry(entry: dict, idx: int, total: int) -> None:
    print(f"\n{'─' * 64}")
    print(f"  [{idx}/{total}]  {entry.get('filename', '?')}")
    print(f"  Language    : {entry.get('source_language', '?')}  "
          f"(lang conf: {entry.get('confidence') or '?'}, "
          f"OCR conf: {entry.get('ocr_confidence') or '?'})")
    print(f"  OCR text    : {entry.get('ocr_text', '')}")
    if entry.get("correct_text") is not None:
        print(f"  Corrected   : {entry['correct_text']}")
    if entry.get("correct_translation") is not None:
        print(f"  Translation : {entry['correct_translation']}")


def review_images(entries: list[dict], review_all: bool) -> int:
    # Uncorrected = no correct_text AND no correct_translation set yet
    pending = entries if review_all else [
        e for e in entries
        if e.get("correct_text") is None and e.get("correct_translation") is None
    ]
    if not pending:
        print("All image entries already have corrections. Use --all to re-review them.")
        return 0

    print(f"\nReviewing {len(pending)} image entr{'y' if len(pending) == 1 else 'ies'} "
          f"({'including already-corrected' if review_all else 'uncorrected only'}).")
    print("Press Ctrl+C at any time to save progress and exit.\n")

    changed = 0
    entry_map = {e["image_hash"]: e for e in entries}

    try:
        for i, entry in enumerate(pending, 1):
            _print_image_entry(entry, i, len(pending))

            new_text = _prompt("Correct OCR text", entry.get("correct_text"))
            new_translation = _prompt("Correct translation", entry.get("correct_translation"))

            if new_text is not None:
                entry["correct_text"] = new_text or None
                changed += 1
            if new_translation is not None:
                entry["correct_translation"] = new_translation or None
                changed += 1

            entry_map[entry["image_hash"]] = entry

    except (KeyboardInterrupt, EOFError):
        print("\n\nInterrupted — saving progress...")

    return changed


# ---------------------------------------------------------------------------
# Text review
# ---------------------------------------------------------------------------

def _print_text_entry(entry: dict, idx: int, total: int) -> None:
    print(f"\n{'─' * 64}")
    print(f"  [{idx}/{total}]  {entry.get('source_language', '?')} → {entry.get('target_language', '?')}"
          f"  (conf: {entry.get('confidence') or '?'}, method: {entry.get('method') or '?'})")
    print(f"  User        : {entry.get('username', '?')}")
    print(f"  Original    : {entry.get('original_text', '')}")
    print(f"  Translation : {entry.get('translated_text', '')}")
    if entry.get("correct_translation") is not None:
        print(f"  Corrected   : {entry['correct_translation']}")


def review_text(entries: list[dict], review_all: bool) -> int:
    pending = entries if review_all else [
        e for e in entries if e.get("correct_translation") is None
    ]
    if not pending:
        print("All text entries already have corrections. Use --all to re-review them.")
        return 0

    print(f"\nReviewing {len(pending)} text entr{'y' if len(pending) == 1 else 'ies'} "
          f"({'including already-corrected' if review_all else 'uncorrected only'}).")
    print("Press Ctrl+C at any time to save progress and exit.\n")

    changed = 0
    entry_map = {e["text_hash"]: e for e in entries}

    try:
        for i, entry in enumerate(pending, 1):
            _print_text_entry(entry, i, len(pending))

            new_translation = _prompt("Correct translation", entry.get("correct_translation"))

            if new_translation is not None:
                entry["correct_translation"] = new_translation or None
                changed += 1

            entry_map[entry["text_hash"]] = entry

    except (KeyboardInterrupt, EOFError):
        print("\n\nInterrupted — saving progress...")

    return changed


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(image_entries: list[dict], text_entries: list[dict]) -> None:
    def _stats(entries: list[dict], *correction_keys: str) -> None:
        total = len(entries)
        corrected = sum(1 for e in entries if any(e.get(k) for k in correction_keys))
        uncorrected = total - corrected
        print(f"  Total       : {total}")
        print(f"  Corrected   : {corrected} ({corrected/total*100:.0f}%)" if total else "  Corrected   : 0")
        print(f"  Uncorrected : {uncorrected} ({uncorrected/total*100:.0f}%)" if total else "  Uncorrected : 0")

    print("\n── Image entries ──────────────────────────────────────")
    if image_entries:
        _stats(image_entries, "correct_text", "correct_translation")
        langs: dict[str, int] = {}
        for e in image_entries:
            langs[e.get("source_language", "unknown")] = langs.get(e.get("source_language", "unknown"), 0) + 1
        print(f"  Languages   : {dict(sorted(langs.items(), key=lambda x: -x[1]))}")
    else:
        print("  No entries found.")

    print("\n── Text entries ───────────────────────────────────────")
    if text_entries:
        _stats(text_entries, "correct_translation")
        langs = {}
        for e in text_entries:
            langs[e.get("source_language", "unknown")] = langs.get(e.get("source_language", "unknown"), 0) + 1
        print(f"  Languages   : {dict(sorted(langs.items(), key=lambda x: -x[1]))}")
    else:
        print("  No entries found.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Review and correct collected training entries.")
    parser.add_argument("--mode", choices=["image", "text"], default="image",
                        help="Which dataset to review (default: image)")
    parser.add_argument("--all", action="store_true", dest="review_all",
                        help="Re-review entries that already have corrections")
    parser.add_argument("--stats", action="store_true",
                        help="Print correction coverage stats and exit")
    args = parser.parse_args()

    image_entries = _load(IMAGE_LABELS)
    text_entries = _load(TEXT_LABELS)

    if args.stats:
        print_stats(image_entries, text_entries)
        return

    if args.mode == "image":
        if not image_entries:
            print(f"No image entries found at {IMAGE_LABELS}")
            sys.exit(1)
        changed = review_images(image_entries, args.review_all)
        if changed:
            _save(IMAGE_LABELS, image_entries)
            print(f"Saved {len(image_entries)} entries to {IMAGE_LABELS}  ({changed} field(s) updated)")
    else:
        if not text_entries:
            print(f"No text entries found at {TEXT_LABELS}")
            sys.exit(1)
        changed = review_text(text_entries, args.review_all)
        if changed:
            _save(TEXT_LABELS, text_entries)
            print(f"Saved {len(text_entries)} entries to {TEXT_LABELS}  ({changed} field(s) updated)")


if __name__ == "__main__":
    main()
