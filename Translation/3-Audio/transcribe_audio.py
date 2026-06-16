"""Audio transcription via HuggingFace Inference API (Whisper).

Transcribes audio to text using openai/whisper-large-v3 via the HF Inference
API. Language detection on the resulting transcript reuses detect.py from the
text pipeline — no separate language model is needed.

See WHISPER_LOCAL.md for how to migrate to a locally-hosted Whisper model
using the same local-first pattern as translate_text.py.

Usage (CLI):
    python transcribe_audio.py audio.ogg
    python transcribe_audio.py audio.ogg --src ko
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import InferenceClient

sys.path.insert(0, str(Path(__file__).parent.parent / "2-Text"))
from detect import detect_language_with_confidence  # noqa: E402

WHISPER_MODEL = "openai/whisper-large-v3"


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="hf-inference", api_key=token)


def transcribe(
    source: bytes | str | Path,
    src_lang: str | None = None,
) -> dict:
    """Transcribe audio to text via HuggingFace Whisper API.

    Args:
        source:   Audio as raw bytes, a file path, or a URL string.
        src_lang: Optional ISO 639-1 language hint. When None, Whisper
                  auto-detects and detect.py refines the language code.

    Returns:
        dict with keys:
            transcript      -- transcribed text string (may be empty)
            source_language -- langdetect-style code, or src_lang if given
            confidence      -- detection confidence [0, 1], or None
            method          -- 'whisper-hf'
    """
    if isinstance(source, Path):
        source = str(source)

    result = _client().automatic_speech_recognition(source, model=WHISPER_MODEL)
    transcript = (result.text or "").strip()

    if src_lang:
        detected, confidence = src_lang, None
    elif transcript:
        detected, confidence = detect_language_with_confidence(transcript)
    else:
        detected, confidence = "unknown", None

    return {
        "transcript": transcript,
        "source_language": detected,
        "confidence": confidence,
        "method": "whisper-hf",
    }


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Transcribe audio to text.")
    parser.add_argument("audio", help="Path to audio file or URL")
    parser.add_argument("--src", default=None, help="Source language hint (e.g. ko, zh)")
    args = parser.parse_args()

    r = transcribe(Path(args.audio) if Path(args.audio).exists() else args.audio,
                   src_lang=args.src)
    print(f"[{r['source_language']} via {r['method']}]")
    print(r["transcript"])
