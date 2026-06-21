"""Tests for synthesize_video.py — MKV with synthesized translated audio.

Validation tests run without assets or network.
Integration tests require the MKV test asset and internet access (gTTS).

Generated MKV files are written to Translation/4-Video/tests/outputs/ and
are kept after the run for manual inspection.

To generate missing assets:
    python Translation/4-Video/tests/assets/gen_video_assets.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3-Audio"))

from synthesize_video import synthesize_video

ASSETS = Path(__file__).parent / "assets"
OUTPUTS = Path(__file__).parent / "outputs"

# EBML magic that opens every MKV/WebM container.
_MKV_MAGIC = b"\x1a\x45\xdf\xa3"


@pytest.fixture(autouse=True, scope="session")
def _ensure_outputs():
    OUTPUTS.mkdir(exist_ok=True)


def _asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        pytest.skip(
            f"Asset not found: {p}  "
            f"(run Translation/4-Video/tests/assets/gen_video_assets.py)"
        )
    return str(p)


# ---------------------------------------------------------------------------
# Validation — no network or assets needed
# ---------------------------------------------------------------------------

class TestSynthesizeVideoValidation:
    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            synthesize_video(b"\x00" * 16, "", "en")

    def test_whitespace_text_raises(self):
        with pytest.raises(ValueError):
            synthesize_video(b"\x00" * 16, "   ", "en")


# ---------------------------------------------------------------------------
# Integration tests — require internet (gTTS) and MKV asset
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSynthesizeVideo:
    """Real gTTS + PyAV calls — require internet access and test asset."""

    def test_returns_bytes(self):
        result = synthesize_video(_asset("test_video_korean.mkv"), "Good morning.", "en")
        assert isinstance(result, bytes) and len(result) > 0

    def test_output_is_mkv(self):
        result = synthesize_video(_asset("test_video_korean.mkv"), "Good morning.", "en")
        assert result[:4] == _MKV_MAGIC

    def test_english_target_saved(self):
        result = synthesize_video(
            _asset("test_video_korean.mkv"),
            "Hello, this is a test of the video synthesis pipeline.",
            "en",
        )
        out = OUTPUTS / "synthesized_video_en.mkv"
        out.write_bytes(result)
        assert out.exists() and out.stat().st_size > 0

    def test_korean_target_saved(self):
        result = synthesize_video(
            _asset("test_video_korean.mkv"),
            "안녕하세요, 이것은 비디오 합성 테스트입니다.",
            "ko",
        )
        out = OUTPUTS / "synthesized_video_ko.mkv"
        out.write_bytes(result)
        assert isinstance(result, bytes) and len(result) > 0

    def test_bytes_source_accepted(self):
        video_bytes = Path(_asset("test_video_korean.mkv")).read_bytes()
        result = synthesize_video(video_bytes, "Hello world.", "en")
        assert result[:4] == _MKV_MAGIC
        (OUTPUTS / "synthesized_video_from_bytes.mkv").write_bytes(result)

    def test_output_differs_from_input(self):
        """Synthesized file should not be byte-identical to the source."""
        source = Path(_asset("test_video_korean.mkv")).read_bytes()
        result = synthesize_video(source, "Good morning.", "en")
        assert result != source
