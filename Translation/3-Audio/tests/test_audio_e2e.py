"""End-to-end integration tests for the audio transcription + translation pipeline.

Calls the real translate_audio() public API (Whisper HF API + translate_text) with no mocks.
Requires test assets in Translation/3-Audio/tests/assets/ and HF_TOKEN in the environment.

To generate missing assets:
    python Translation/3-Audio/tests/assets/gen_audio_assets.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1-Text"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from translate_audio import translate_audio


pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
needs_token = pytest.mark.skipif(not _HAS_TOKEN, reason="HF_TOKEN not set")


def _asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        pytest.skip(
            f"Asset not found: {p}  "
            f"(run Translation/3-Audio/tests/assets/gen_audio_assets.py)"
        )
    return str(p)


@needs_token
class TestAudioE2E:
    """Real Whisper API + translation calls — require HF_TOKEN."""

    def test_result_has_required_keys(self):
        r = translate_audio(_asset("test_audio_korean.ogg"),
                            from_lang=None, to_lang=None,
                            filename="test_audio_korean.ogg", username="pytest")
        for key in ("original_text", "translated_text", "source_language",
                    "confidence", "method", "collected"):
            assert key in r

    def test_korean_audio_transcribed(self):
        r = translate_audio(_asset("test_audio_korean.ogg"),
                            from_lang=None, to_lang=None,
                            filename="test_audio_korean.ogg", username="pytest")
        assert r["original_text"].strip(), "Whisper returned empty transcript for Korean audio"
        assert r["source_language"] == "ko"
        assert r["method"] == "whisper-hf"

    def test_korean_audio_translated_to_english(self):
        r = translate_audio(_asset("test_audio_korean.ogg"),
                            from_lang=None, to_lang=None,
                            filename="test_audio_korean.ogg", username="pytest")
        assert r["translated_text"].strip()

    def test_chinese_audio_transcribed(self):
        r = translate_audio(_asset("test_audio_chinese.ogg"),
                            from_lang=None, to_lang=None,
                            filename="test_audio_chinese.ogg", username="pytest")
        assert r["original_text"].strip()
        assert r["source_language"] in ("zh-cn", "zh")

    def test_from_lang_hint_used(self):
        r = translate_audio(_asset("test_audio_korean.ogg"),
                            from_lang="ko", to_lang=None,
                            filename="test_audio_korean.ogg", username="pytest")
        assert r["source_language"] == "ko"

    def test_empty_audio_returns_empty_result(self):
        """Verify pipeline handles empty-transcript case gracefully (no crash)."""
        from unittest.mock import patch, MagicMock
        mock_result = MagicMock()
        mock_result.text = ""
        with patch("transcribe_audio._client") as mock_client:
            mock_client.return_value.automatic_speech_recognition.return_value = mock_result
            r = translate_audio(_asset("test_audio_korean.ogg"),
                                from_lang=None, to_lang=None,
                                filename="test_audio_korean.ogg", username="pytest")
        assert r["original_text"] == ""
        assert r["collected"] is False
