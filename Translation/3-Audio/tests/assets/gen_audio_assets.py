"""Generate test audio assets for the 3-Audio integration tests.

Run from repo root:
    python Translation/3-Audio/tests/assets/gen_audio_assets.py

Produces OGG (MP3) files via gTTS for Korean, Chinese, and Japanese phrases.
Requires internet access (gTTS). No HF_TOKEN needed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

SAMPLES = [
    ("korean",   "ko",    "안녕하세요, 오늘 날씨가 좋네요"),
    ("chinese",  "zh-CN", "早上好，今天天气很好"),
    ("japanese", "ja",    "おはようございます"),
]


def main() -> None:
    from gtts import gTTS
    for label, lang, text in SAMPLES:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        out = HERE / f"test_audio_{label}.ogg"
        out.write_bytes(buf.read())
        print(f"Wrote {out.name}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
