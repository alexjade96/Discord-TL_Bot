"""End-to-end integration tests for the video translation pipeline.

Calls the real translate_video() public API (PyAV audio extraction + Whisper + translate_text)
with no mocks. Requires test assets in Translation/4-Video/tests/assets/ and HF_TOKEN.

To generate missing assets:
    python Translation/4-Video/tests/assets/gen_video_assets.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1-Text"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3-Audio"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "0-Data" / "Video" / "training"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from translate_video import translate_video
from extract_audio import extract_audio


pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
needs_token = pytest.mark.skipif(not _HAS_TOKEN, reason="HF_TOKEN not set")


def _asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        pytest.skip(
            f"Asset not found: {p}  "
            f"(run Translation/4-Video/tests/assets/gen_video_assets.py)"
        )
    return str(p)


class TestExtractAudioE2E:
    """extract_audio() with a real MKV file — no HF_TOKEN needed."""

    def test_extract_returns_wav_bytes(self):
        wav = extract_audio(_asset("test_video_korean.mkv"))
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_extracted_wav_non_empty(self):
        wav = extract_audio(_asset("test_video_korean.mkv"))
        assert len(wav) > 44

    def test_path_object_accepted(self):
        p = Path(_asset("test_video_korean.mkv"))
        wav = extract_audio(p)
        assert wav[:4] == b"RIFF"


@needs_token
class TestVideoE2E:
    """Full pipeline: extract → transcribe → translate. Requires HF_TOKEN."""

    def test_result_has_required_keys(self):
        r = translate_video(_asset("test_video_korean.mkv"),
                            filename="test_video_korean.mkv", username="pytest")
        for key in ("original_text", "translated_text", "source_language",
                    "confidence", "method", "collected"):
            assert key in r

    def test_korean_video_transcribed(self):
        r = translate_video(_asset("test_video_korean.mkv"),
                            filename="test_video_korean.mkv", username="pytest")
        assert r["original_text"].strip(), "Whisper returned empty transcript"
        assert r["source_language"] == "ko"

    def test_korean_video_translated_to_english(self):
        r = translate_video(_asset("test_video_korean.mkv"),
                            filename="test_video_korean.mkv", username="pytest")
        assert r["translated_text"].strip()

    def test_from_lang_hint_propagated(self):
        r = translate_video(_asset("test_video_korean.mkv"),
                            from_lang="ko",
                            filename="test_video_korean.mkv", username="pytest")
        assert r["source_language"] == "ko"

    def test_method_is_whisper_hf(self):
        r = translate_video(_asset("test_video_korean.mkv"),
                            filename="test_video_korean.mkv", username="pytest")
        assert r["method"] == "whisper-hf"
