"""Text-to-speech synthesis for the audio translation pipeline.

Converts translated text to an MP3 audio file using gTTS (Google TTS).
This is the output side of the audio pipeline:

    source audio → transcribe → translate → synthesize → Discord file

Usage (CLI):
    python synthesize_audio.py "안녕하세요" --lang ko
    python synthesize_audio.py "Hello world" --lang en --out out.mp3
"""

from __future__ import annotations

import io
from pathlib import Path

# gTTS uses BCP-47 codes; our pipeline uses lowercase ISO 639-1.
# Most codes are identical; exceptions are listed here.
_GTTS_CODE_MAP: dict[str, str] = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "zh":    "zh-CN",
}


def _to_gtts_lang(lang: str) -> str:
    """Normalise an internal language code to a gTTS-compatible BCP-47 code."""
    return _GTTS_CODE_MAP.get(lang.lower(), lang.lower())


def synthesize(text: str, lang: str) -> bytes:
    """Convert translated text to MP3 audio via gTTS.

    Args:
        text: The text to speak.  Must be non-empty.
        lang: Language code accepted by gTTS (e.g. 'en', 'ko', 'zh-CN', 'ja').
              Internal codes like 'zh-cn' are normalised automatically.

    Returns:
        MP3 audio bytes suitable for sending as a Discord file attachment.

    Raises:
        ValueError:   If text is empty or whitespace-only.
        gtts.gTTSError: If the Google TTS request fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=_to_gtts_lang(lang)).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    import argparse
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Synthesize speech from text via gTTS.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--out", default=None, help="Output MP3 path (default: synthesized.mp3)")
    args = parser.parse_args()

    audio = synthesize(args.text, args.lang)
    out_path = Path(args.out) if args.out else Path("synthesized.mp3")
    out_path.write_bytes(audio)
    print(f"Wrote {len(audio):,} bytes → {out_path}")
