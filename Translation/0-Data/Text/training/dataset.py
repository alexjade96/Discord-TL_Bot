"""Build a translation dataset from collected text submissions.

Reads text_submissions.jsonl, prefers correct_translation over translated_text
when set (same principle as image dataset.py preferring correct_text over ocr_text),
and exports source/target pairs as JSONL files consumable by train.py.

Two pair directions are produced:
  - mul-en pairs: (original_text → english)  used to fine-tune opus-mt-mul-en
  - en-{tgt} pairs: (english → original_text) synthesised by flipping mul-en pairs,
    used to fine-tune the reverse models per target language

Usage:
    python dataset.py                   # build all pairs from data/
    python dataset.py --split 0.9       # 90/10 train/val split
    python dataset.py --src ko          # only Korean source pairs
    python dataset.py --tgt fr          # only en→fr reverse pairs
    python dataset.py --stats           # print coverage and exit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_FILE = DATA_DIR / "text_submissions.jsonl"
DEFAULT_OUT = Path(__file__).parent / "dataset"


def load_submissions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _best_translation(entry: dict) -> str:
    """Return correct_translation if set, else translated_text."""
    return entry.get("correct_translation") or entry.get("translated_text") or ""


def build_pairs(submissions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert submissions into mul-en and en-{tgt} pair lists.

    mul-en pairs: (original_text, english_translation) from submissions
        where target_language == 'en'.
    en-{tgt} pairs: synthesised by reversing mul-en pairs — each
        (english → original_text) entry is tagged with the source language
        as the effective target.

    Returns (mul_en_pairs, reverse_pairs).
    """
    mul_en: list[dict] = []
    reverse: list[dict] = []

    corrected = 0
    for entry in submissions:
        tgt = entry.get("target_language", "en")
        src = entry.get("source_language", "unknown")
        original = (entry.get("original_text") or "").strip()
        translation = _best_translation(entry).strip()

        if not original or not translation:
            continue
        if src in ("unknown", "en"):
            continue

        if tgt == "en":
            if entry.get("correct_translation"):
                corrected += 1
            mul_en.append({
                "source": original,
                "target": translation,
                "src_lang": src,
                "tgt_lang": "en",
            })
            # Synthesise reverse pair for en→src fine-tuning
            reverse.append({
                "source": translation,
                "target": original,
                "src_lang": "en",
                "tgt_lang": src.split("-")[0],
            })

    print(f"  mul-en pairs   : {len(mul_en)} ({corrected} used correct_translation)")
    print(f"  en-{{tgt}} pairs : {len(reverse)} (synthesised from mul-en by reversal)")
    return mul_en, reverse


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def split_and_write(
    pairs: list[dict],
    out_prefix: Path,
    split: float | None,
    seed: int = 42,
) -> None:
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    if split:
        idx = int(len(shuffled) * split)
        train, val = shuffled[:idx], shuffled[idx:]
        _write_jsonl(Path(f"{out_prefix}_train.jsonl"), train)
        _write_jsonl(Path(f"{out_prefix}_val.jsonl"), val)
        print(f"  → {out_prefix}_train.jsonl  ({len(train)} pairs)")
        print(f"  → {out_prefix}_val.jsonl    ({len(val)} pairs)")
    else:
        _write_jsonl(Path(f"{out_prefix}.jsonl"), shuffled)
        print(f"  → {out_prefix}.jsonl  ({len(shuffled)} pairs)")


def print_stats(submissions: list[dict]) -> None:
    total = len(submissions)
    if not total:
        print("\n  No submissions yet.")
        return

    corrected = sum(1 for e in submissions if e.get("correct_translation"))
    src_langs: dict[str, int] = {}
    tgt_langs: dict[str, int] = {}
    for e in submissions:
        src = e.get("source_language", "unknown")
        tgt = e.get("target_language", "en")
        src_langs[src] = src_langs.get(src, 0) + 1
        tgt_langs[tgt] = tgt_langs.get(tgt, 0) + 1

    print(f"\n  Total submissions  : {total}")
    print(f"  Corrected          : {corrected} ({corrected/total*100:.0f}%)")
    print(f"  Source languages   : {dict(sorted(src_langs.items(), key=lambda x: -x[1]))}")
    print(f"  Target languages   : {dict(sorted(tgt_langs.items(), key=lambda x: -x[1]))}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text translation dataset.")
    parser.add_argument("--data", default=str(SUBMISSIONS_FILE), help="Path to text_submissions.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path prefix")
    parser.add_argument("--split", type=float, default=None, help="Train/val split ratio (e.g. 0.9)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", default=None, help="Filter by source language code (e.g. ko)")
    parser.add_argument("--tgt", default=None, help="Filter reverse pairs by target language code")
    parser.add_argument("--stats", action="store_true", help="Print dataset stats and exit")
    args = parser.parse_args()

    submissions = load_submissions(Path(args.data))

    if args.stats:
        print_stats(submissions)
        return

    if args.src:
        submissions = [e for e in submissions if e.get("source_language", "").split("-")[0] == args.src]
        print(f"Filtered to src={args.src}: {len(submissions)} entries")

    if not submissions:
        print(f"No submissions found at {args.data}")
        print("Submissions are collected automatically when the bot processes /translate commands.")
        return

    print(f"\nLoaded {len(submissions)} submissions from {args.data}")
    mul_en_pairs, reverse_pairs = build_pairs(submissions)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.tgt:
        print("\n[mul-en]")
        if mul_en_pairs:
            split_and_write(mul_en_pairs, out / "mul-en", args.split, args.seed)
        else:
            print("  No mul-en pairs — skipping.")

    if not args.src:
        print("\n[en-{tgt}]")
        if args.tgt:
            reverse_pairs = [p for p in reverse_pairs if p["tgt_lang"] == args.tgt]
        # Group by target language and write one file per language
        by_tgt: dict[str, list[dict]] = {}
        for pair in reverse_pairs:
            by_tgt.setdefault(pair["tgt_lang"], []).append(pair)
        if by_tgt:
            for lang, pairs in sorted(by_tgt.items()):
                print(f"  en-{lang}:")
                split_and_write(pairs, out / f"en-{lang}", args.split, args.seed)
        else:
            print("  No reverse pairs — skipping.")


if __name__ == "__main__":
    main()
