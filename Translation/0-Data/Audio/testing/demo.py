"""Full audio pipeline demonstration using gTTS-synthesized speech.

Synthesizes sample phrases in Korean, Chinese, Japanese, and English via
gTTS, runs each through transcribe → translate → collect, then summarises
the resulting dataset. Requires internet access (gTTS + HF Whisper API).

Usage:
    python demo.py
    python demo.py --no-collect     # skip saving to data/
    python demo.py --tgt fr         # translate to French instead of English
    python demo.py --save-audio     # write generated MP3s to demo_output/
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

# Force UTF-8 output so CJK characters print correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(HERE.parent.parent.parent.parent / ".env")

# Pipeline imports
sys.path.insert(0, str(HERE.parent.parent.parent / "3-Audio"))
sys.path.insert(0, str(HERE.parent / "training"))

# fmt: off
SAMPLES = [
    {
        "label":    "Korean",
        "lang":     "ko",
        "texts":    ["안녕하세요", "감사합니다", "오늘 날씨가 좋네요"],
        "expected": "Korean → English",
    },
    {
        "label":    "Chinese (Simplified)",
        "lang":     "zh-CN",
        "texts":    ["早上好", "谢谢你", "今天天气很好"],
        "expected": "Chinese → English",
    },
    {
        "label":    "Japanese",
        "lang":     "ja",
        "texts":    ["おはようございます", "ありがとう", "東京は美しい"],
        "expected": "Japanese → English",
    },
    {
        "label":    "English",
        "lang":     "en",
        "texts":    ["Good morning", "Hello world", "How are you today?"],
        "expected": "passthrough",
    },
]
# fmt: on


def synthesize_mp3(text: str, lang: str) -> bytes:
    """Return gTTS MP3 bytes for *text* in *lang* (BCP-47 code)."""
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def run_demo(collect: bool = True, tgt: str | None = None, save_audio: bool = False) -> None:
    from translate_audio import translate_audio
    from collect_audio import dataset_stats

    output_dir = HERE.parent / "data" / "demo_output"
    if save_audio:
        output_dir.mkdir(exist_ok=True)

    total_passed = 0
    total_failed = 0
    tgt_label = tgt or "en"

    for group in SAMPLES:
        print(f"\n{'-' * 60}")
        print(f"  {group['label']}  (expected: {group['expected']})")
        print(f"{'-' * 60}")

        for text in group["texts"]:
            try:
                audio_bytes = synthesize_mp3(text, group["lang"])

                if save_audio:
                    slug = text[:20].replace(" ", "_").replace("/", "")
                    out_path = output_dir / f"{group['label'].replace(' ', '_')}_{slug}.mp3"
                    out_path.write_bytes(audio_bytes)

                result = translate_audio(
                    audio_bytes,
                    to_lang=tgt,
                    filename=f"demo_{group['lang']}.mp3",
                    username="demo",
                )
                transcript = result["original_text"]
                translated = result["translated_text"]
                lang_code  = result["source_language"]
                conf       = result["confidence"]
                method     = result["method"]

                conf_str = f" {conf*100:.0f}%" if conf is not None else ""
                print(f"\n  Input      : {text!r}")
                print(f"  Transcript : {transcript!r}")
                print(f"  Output     : {translated!r}")
                print(f"  Language   : {lang_code}{conf_str}  [{method}]")

                if collect:
                    status = "saved" if result.get("collected") else "duplicate (skipped)"
                    print(f"  Collected  : {status}")

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
    parser = argparse.ArgumentParser(description="Full audio pipeline demo.")
    parser.add_argument("--no-collect", action="store_true", help="Skip saving to data/")
    parser.add_argument("--tgt", default=None, help="Target language code (default: en)")
    parser.add_argument("--save-audio", action="store_true", help="Write generated MP3s to demo_output/")
    args = parser.parse_args()

    print("\nTL-Bot Audio Pipeline Demo")
    print("Synthesizing speech → transcribe → translate → collect\n")
    print("Note: Requires internet access for gTTS synthesis and HF Whisper API.\n")

    run_demo(collect=not args.no_collect, tgt=args.tgt, save_audio=args.save_audio)
