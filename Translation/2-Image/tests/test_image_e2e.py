"""End-to-end integration tests for the image translation pipeline.

Calls the real translate_image() public API (EasyOCR + translate_text) with no mocks.
Requires test assets in Translation/2-Image/tests/assets/.
Tests are skipped when the asset files are absent or HF_TOKEN is missing.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1-Text"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from translate_image import translate_image


pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
needs_token = pytest.mark.skipif(not _HAS_TOKEN, reason="HF_TOKEN not set")


def _asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        pytest.skip(f"Asset not found: {p}  (run Translation/3-Audio/tests/assets/gen_audio_assets.py)")
    return str(p)


@needs_token
class TestImageE2E:
    """Real EasyOCR + HF API calls — require HF_TOKEN."""

    def test_result_structure(self):
        r = translate_image(_asset("test_image.png"))
        auto = r["auto"]
        for key in ("original_text", "translated_text", "source_language",
                    "confidence", "ocr_confidence", "method", "score"):
            assert key in auto

    def test_hint_is_none_without_hint_lang(self):
        r = translate_image(_asset("test_image.png"))
        assert r["hint"] is None

    def test_ocr_extracts_text(self):
        r = translate_image(_asset("test_image.png"))
        assert r["auto"]["original_text"].strip()

    def test_translation_returned(self):
        r = translate_image(_asset("test_image.png"))
        assert r["auto"]["translated_text"].strip()

    def test_second_image_pipeline(self):
        r = translate_image(_asset("test_image_2.png"))
        auto = r["auto"]
        assert auto["original_text"].strip()
        assert auto["translated_text"].strip()

    def test_hint_pass_runs_when_hint_given(self):
        r = translate_image(_asset("test_image.png"), hint_lang="zh")
        assert r["hint"] is not None
        assert r["hint"]["method"] != "none"
