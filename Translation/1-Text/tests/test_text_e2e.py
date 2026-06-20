"""End-to-end integration tests for the text translation pipeline.

Calls the real translate_text() public API with no mocks. Tests that need the
HF Inference API are skipped automatically when HF_TOKEN is absent.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from translate_text import translate_text


pytestmark = pytest.mark.integration

_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
needs_token = pytest.mark.skipif(not _HAS_TOKEN, reason="HF_TOKEN not set")


class TestTextPassthrough:
    """These run without HF_TOKEN — no API call is made."""

    def test_empty_string_no_translation(self):
        r = translate_text("")
        assert r["method"] == "none"
        assert r["translated_text"] == ""

    def test_english_returns_passthrough(self):
        r = translate_text("Hello, how are you?")
        assert r["method"] == "passthrough"
        assert r["translated_text"] == "Hello, how are you?"

    def test_same_src_and_tgt_language_passthrough(self):
        r = translate_text("안녕하세요", src_lang="ko", tgt_lang="ko")
        assert r["method"] == "passthrough"
        assert r["source_language"] == "ko"

    def test_result_has_all_required_keys(self):
        r = translate_text("Hello")
        for key in ("translated_text", "source_language", "confidence", "method"):
            assert key in r


@needs_token
class TestTextE2E:
    """Real API calls — require HF_TOKEN."""

    def test_chinese_to_english(self):
        r = translate_text("早上好")
        assert r["source_language"] in ("zh-cn", "zh")
        assert r["translated_text"]
        assert r["method"] not in ("none", "passthrough")

    def test_japanese_to_english(self):
        r = translate_text("おはようございます")
        assert r["source_language"] == "ja"
        assert r["translated_text"]
        assert r["method"] not in ("none", "passthrough")

    def test_korean_to_english(self):
        r = translate_text("안녕하세요")
        assert r["source_language"] == "ko"
        assert r["translated_text"]
        assert r["method"] not in ("none", "passthrough")

    def test_english_to_french(self):
        r = translate_text("Good morning", tgt_lang="fr")
        assert r["translated_text"]
        assert r["method"] not in ("none", "passthrough")

    def test_mixed_cjk_english_segmented(self):
        r = translate_text("coach hug 赢家的视角")
        assert r["translated_text"]
        assert r["method"] == "opus-mt-segmented"
