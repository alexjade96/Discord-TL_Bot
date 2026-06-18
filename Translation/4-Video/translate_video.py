"""Full video pipeline: extract audio → transcribe → translate → collect.

Mirrors translate_audio.py. Accepts a URL, local path, or bytes, strips the
audio track via ffmpeg, transcribes via Whisper (HF API), translates the
transcript via the text pipeline, and non-fatally collects the submission.

Usage (CLI):
    python translate_video.py video.mp4
    python translate_video.py clip.mp4 --from ko --to en
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "1-Text"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3-Audio"))
sys.path.insert(0, str(Path(__file__).parent.parent / "0-Data" / "Video" / "training"))

from extract_audio import extract_audio          # noqa: E402 (same package)
from transcribe_audio import transcribe          # noqa: E402
from translate_text import translate_text        # noqa: E402


def translate_video(
    source: bytes | str | Path,
    from_lang: str | None = None,
    to_lang: str | None = None,
    filename: str = "",
    username: str = "",
) -> dict:
    """Extract audio from a video, transcribe, and translate the transcript.

    Args:
        source:    Video bytes, local file path, or URL string.
        from_lang: Language hint for Whisper and translation.
        to_lang:   Target language (default: English).
        filename:  Original Discord filename, used for collection logging.
        username:  Discord username, used for collection logging.

    Returns:
        dict with keys:
            original_text    -- raw Whisper transcript
            translated_text  -- translated transcript
            source_language  -- detected language code
            confidence       -- detection confidence or None
            method           -- transcription method ('whisper-hf')
            collected        -- True if submission was saved, False otherwise
    """
    audio_bytes = extract_audio(source)

    transcription = transcribe(audio_bytes, src_lang=from_lang)
    transcript = transcription["transcript"]
    src_lang = transcription["source_language"]
    confidence = transcription["confidence"]
    method = transcription["method"]

    if not transcript:
        return {
            "original_text": "",
            "translated_text": "",
            "source_language": src_lang,
            "confidence": confidence,
            "method": method,
            "collected": False,
        }

    translation = translate_text(transcript, src_lang=src_lang, tgt_lang=to_lang)
    translated = translation["translated_text"]
    translate_method = translation["method"]

    collected = False
    if translate_method not in ("none", "passthrough"):
        try:
            from collect_video import save_submission
            collected = save_submission(
                transcript=transcript,
                translated_text=translated,
                source_language=src_lang,
                target_language=to_lang or "en",
                confidence=confidence,
                method=method,
                username=username,
                filename=filename,
            )
        except Exception:
            pass

    return {
        "original_text": transcript,
        "translated_text": translated,
        "source_language": src_lang,
        "confidence": confidence,
        "method": method,
        "collected": collected,
    }


if __name__ == "__main__":
    import argparse
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Extract audio from video, transcribe, and translate.")
    parser.add_argument("source", help="Video file path or URL")
    parser.add_argument("--from", dest="src", default=None, help="Source language hint")
    parser.add_argument("--to", dest="tgt", default=None, help="Target language (default: en)")
    args = parser.parse_args()

    r = translate_video(args.source, from_lang=args.src, to_lang=args.tgt)
    print(f"[{r['source_language']} → {args.tgt or 'en'} via {r['method']}]")
    print(f"Transcript  : {r['original_text']}")
    print(f"Translation : {r['translated_text']}")
