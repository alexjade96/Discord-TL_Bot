"""Full text pipeline demonstration.

Runs a fixed set of sample source texts through translate → collect → report,
mirroring the image pipeline's demo.py. Useful for validating the full text
pipeline after changes to translate_text.py or collect_text.py.

Usage:
    python demo.py
    python demo.py --no-collect    # skip saving to data/
    python demo.py --tgt fr        # translate to French instead of English
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 output so CJK characters print correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(HERE.parent.parent.parent.parent / ".env")

# Pipeline imports
sys.path.insert(0, str(HERE.parent.parent.parent / "2-Text"))
sys.path.insert(0, str(HERE.parent / "training"))

# fmt: off
SAMPLES = [
    {
        "label":    "Korean",
        "texts":    ["안녕하세요", "감사합니다", "서울에 오신 것을 환영합니다", "오늘 날씨가 좋네요"],
        "expected": "Korean → English",
    },
    {
        "label":    "Chinese (Simplified)",
        "texts":    ["早上好", "谢谢你", "你好世界", "今天天气很好"],
        "expected": "Chinese → English",
    },
    {
        "label":    "Japanese",
        "texts":    ["おはようございます", "ありがとう", "東京は美しい"],
        "expected": "Japanese → English",
    },
    {
        "label":    "Mixed (English + Korean)",
        "texts":    ["hello 안녕 world", "Game over 게임 오버", "Level up 레벨 업"],
        "expected": "Korean segments → English",
    },
    {
        "label":    "English",
        "texts":    ["Good morning", "Hello world", "How are you today?"],
        "expected": "passthrough",
    },
]
# fmt: on


def run_demo(collect: bool = True, tgt: str | None = None) -> None:
    from translate_text import translate_text
    from collect_text import save_submission, dataset_stats

    total_passed = 0
    total_failed = 0
    tgt_label = tgt or "en"

    for group in SAMPLES:
        print(f"\n{'-' * 60}")
        print(f"  {group['label']}  (expected: {group['expected']})")
        print(f"{'-' * 60}")

        for text in group["texts"]:
            try:
                result = translate_text(text, tgt_lang=tgt)
                method = result["method"]
                lang = result["source_language"]
                conf = result.get("confidence")
                translated = result["translated_text"]

                conf_str = f" {conf*100:.0f}%" if conf is not None else ""
                print(f"\n  Input      : {text!r}")
                print(f"  Output     : {translated!r}")
                print(f"  Language   : {lang}{conf_str}  [{method}]")

                if collect and method not in ("none", "passthrough"):
                    saved = save_submission(
                        original_text=text,
                        translated_text=translated,
                        source_language=lang,
                        target_language=tgt_label,
                        confidence=conf,
                        method=method,
                        username="demo",
                    )
                    print(f"  Collected  : {'saved' if saved else 'duplicate (skipped)'}")

                total_passed += 1

            except Exception as e:
                print(f"\n  Input  : {text!r}")
                print(f"  ERROR  : {e}")
                total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {total_passed} passed / {total_failed} failed")

    if collect:
        stats = dataset_stats()
        print(f"\n  Dataset after demo:")
        print(f"    Total collected : {stats['total']} submissions")
        print(f"    Languages       : {stats['languages']}")
        print(f"    Location        : {stats['file']}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full text pipeline demo.")
    parser.add_argument("--no-collect", action="store_true", help="Skip saving to data/")
    parser.add_argument("--tgt", default=None, help="Target language code (default: en)")
    args = parser.parse_args()

    print("\nTL-Bot Text Pipeline Demo")
    print("Sample texts → translate → collect → report\n")

    run_demo(collect=not args.no_collect, tgt=args.tgt)
