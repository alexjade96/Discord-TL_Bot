"""Tests for synthesize_audio.py — MP3 synthesis via gTTS.

Pure-function tests (code normalisation, empty-text validation) run offline.
Integration tests require internet access (gTTS) and are marked accordingly.

Generated MP3 files are written to Translation/3-Audio/tests/outputs/ and
are kept after the run for manual inspection.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesize_audio import synthesize, _to_gtts_lang

OUTPUTS = Path(__file__).parent / "outputs"


@pytest.fixture(autouse=True, scope="session")
def _ensure_outputs():
    OUTPUTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# _to_gtts_lang — pure function, no network
# ---------------------------------------------------------------------------

class TestToGttsLang:
    def test_en_unchanged(self):
        assert _to_gtts_lang("en") == "en"

    def test_ko_unchanged(self):
        assert _to_gtts_lang("ko") == "ko"

    def test_ja_unchanged(self):
        assert _to_gtts_lang("ja") == "ja"

    def test_zh_cn_mapped(self):
        assert _to_gtts_lang("zh-cn") == "zh-CN"

    def test_zh_tw_mapped(self):
        assert _to_gtts_lang("zh-tw") == "zh-TW"

    def test_zh_mapped_to_simplified(self):
        assert _to_gtts_lang("zh") == "zh-CN"

    def test_uppercase_input_normalised(self):
        assert _to_gtts_lang("ZH-CN") == "zh-CN"

    def test_unknown_code_passed_through(self):
        assert _to_gtts_lang("fr") == "fr"


# ---------------------------------------------------------------------------
# synthesize() validation — no network
# ---------------------------------------------------------------------------

class TestSynthesizeValidation:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            synthesize("", "en")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            synthesize("   ", "en")


# ---------------------------------------------------------------------------
# Integration tests — require internet (gTTS)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSynthesizeAudio:
    """Real gTTS calls — require internet access."""

    def test_english_returns_bytes(self):
        result = synthesize("Good morning", "en")
        assert isinstance(result, bytes) and len(result) > 0

    def test_output_has_mp3_magic(self):
        result = synthesize("Good morning", "en")
        assert result[:3] == b"ID3" or result[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")

    def test_english_output_saved(self):
        result = synthesize("The quick brown fox jumps over the lazy dog.", "en")
        out = OUTPUTS / "synthesized_en.mp3"
        out.write_bytes(result)
        assert out.exists() and out.stat().st_size > 0

    def test_korean_synthesis(self):
        result = synthesize("안녕하세요, 오늘 날씨가 좋네요.", "ko")
        assert isinstance(result, bytes) and len(result) > 0
        (OUTPUTS / "synthesized_ko.mp3").write_bytes(result)

    def test_chinese_synthesis(self):
        result = synthesize("早上好，今天天气很好。", "zh-cn")
        assert isinstance(result, bytes) and len(result) > 0
        (OUTPUTS / "synthesized_zh.mp3").write_bytes(result)

    def test_japanese_synthesis(self):
        result = synthesize("おはようございます。", "ja")
        assert isinstance(result, bytes) and len(result) > 0
        (OUTPUTS / "synthesized_ja.mp3").write_bytes(result)

    def test_french_synthesis(self):
        result = synthesize("Bonjour, comment allez-vous?", "fr")
        assert isinstance(result, bytes) and len(result) > 0
        (OUTPUTS / "synthesized_fr.mp3").write_bytes(result)

    def test_different_texts_produce_different_output(self):
        a = synthesize("Hello", "en")
        b = synthesize("Goodbye", "en")
        assert a != b
