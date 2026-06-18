"""Full video pipeline demonstration using gTTS audio wrapped in a Matroska container.

Synthesizes sample phrases via gTTS, wraps each in an MKV container using PyAV
(stream copy, no transcode), runs through extract_audio → transcribe →
translate → collect, then summarises the resulting dataset.

Requires internet access (gTTS + HF Whisper API). No system ffmpeg needed.

Usage:
    python demo.py
    python demo.py --no-collect     # skip saving to data/
    python demo.py --tgt fr         # translate to French instead of English
    python demo.py --save-videos    # write generated MKVs to demo_output/
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import av

# Force UTF-8 output so CJK characters print correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(HERE.parent.parent.parent.parent / ".env")

# Pipeline imports
sys.path.insert(0, str(HERE.parent.parent.parent / "4-Video"))
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


def wrap_audio_in_mkv(audio_bytes: bytes) -> bytes:
    """Wrap MP3 audio bytes into a Matroska container using PyAV (decode → Opus encode at 48kHz)."""
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as vf:
        out_path = vf.name
    try:
        with av.open(io.BytesIO(audio_bytes)) as in_c:
            in_stream = next(s for s in in_c.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
            with av.open(out_path, mode="w", format="matroska") as out_c:
                out_stream = out_c.add_stream("libopus", rate=48000)
                for frame in in_c.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        resampled.pts = None
                        for packet in out_stream.encode(resampled):
                            out_c.mux(packet)
                for resampled in resampler.resample(None):
                    resampled.pts = None
                    for packet in out_stream.encode(resampled):
                        out_c.mux(packet)
                for packet in out_stream.encode(None):
                    out_c.mux(packet)
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


def run_demo(collect: bool = True, tgt: str | None = None, save_videos: bool = False) -> None:
    from translate_video import translate_video
    from collect_video import dataset_stats

    output_dir = HERE.parent / "data" / "demo_output"
    if save_videos:
        output_dir.mkdir(exist_ok=True)

    total_passed = 0
    total_failed = 0

    for group in SAMPLES:
        print(f"\n{'-' * 60}")
        print(f"  {group['label']}  (expected: {group['expected']})")
        print(f"{'-' * 60}")

        for text in group["texts"]:
            try:
                audio_bytes = synthesize_mp3(text, group["lang"])
                video_bytes = wrap_audio_in_mkv(audio_bytes)

                if save_videos:
                    slug = text[:20].replace(" ", "_").replace("/", "")
                    out_path = output_dir / f"{group['label'].replace(' ', '_')}_{slug}.mkv"
                    out_path.write_bytes(video_bytes)

                result = translate_video(
                    video_bytes,
                    to_lang=tgt,
                    filename=f"demo_{group['lang']}.mkv",
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
    parser = argparse.ArgumentParser(description="Full video pipeline demo.")
    parser.add_argument("--no-collect", action="store_true", help="Skip saving to data/")
    parser.add_argument("--tgt", default=None, help="Target language code (default: en)")
    parser.add_argument("--save-videos", action="store_true", help="Write generated MKVs to demo_output/")
    args = parser.parse_args()

    print("\nTL-Bot Video Pipeline Demo")
    print("Synthesizing speech → wrap in MKV → extract audio → transcribe → translate → collect\n")
    print("Note: Requires internet access (gTTS + HF Whisper API).\n")

    run_demo(collect=not args.no_collect, tgt=args.tgt, save_videos=args.save_videos)
