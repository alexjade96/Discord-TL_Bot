"""Tests for synthesize_image.py — image synthesis with translated text.

synthesize_text_to_image tests are pure PIL and always run (no network, no assets).
synthesize_image tests with fake segments also run always.
The real-asset OCR test is marked integration (requires EasyOCR + test_image.png).

Generated PNG files are written to Translation/2-Image/tests/outputs/ and
are kept after the run for manual inspection.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1-Text"))

from synthesize_image import (
    synthesize_image,
    synthesize_text_to_image,
    _CANVAS_WIDTH,
    _CANVAS_PADDING,
    _CANVAS_BG_DARK,
    _CANVAS_BG_LIGHT,
)

ASSETS = Path(__file__).parent / "assets"
OUTPUTS = Path(__file__).parent / "outputs"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Minimal OCR segment over a small region of a blank image.
_FAKE_SEGMENTS = [
    {
        "text": "Hello",
        "confidence": 0.95,
        "bbox": [[10, 10], [110, 10], [110, 40], [10, 40]],
    }
]


@pytest.fixture(autouse=True, scope="session")
def _ensure_outputs():
    OUTPUTS.mkdir(exist_ok=True)


def _asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        pytest.skip(f"Asset not found: {p}")
    return str(p)


def _blank_png(width: int = 400, height: int = 100,
               color: tuple = (240, 240, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _img_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def _img_pixel(data: bytes, xy: tuple[int, int] = (0, 0)) -> tuple[int, int, int]:
    return Image.open(io.BytesIO(data)).convert("RGB").getpixel(xy)


# ---------------------------------------------------------------------------
# synthesize_text_to_image — pure PIL, no network, always runs
# ---------------------------------------------------------------------------

class TestSynthesizeTextToImage:
    def test_returns_png_bytes(self):
        assert synthesize_text_to_image("Good morning", "en")[:8] == _PNG_MAGIC

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            synthesize_text_to_image("", "en")

    def test_whitespace_text_raises(self):
        with pytest.raises(ValueError):
            synthesize_text_to_image("   ", "en")

    def test_canvas_width_matches_constant(self):
        w, _ = _img_size(synthesize_text_to_image("Hello", "en"))
        assert w == _CANVAS_WIDTH

    def test_light_mode_background(self):
        px = _img_pixel(synthesize_text_to_image("Hello", "en", dark=False))
        assert px == _CANVAS_BG_LIGHT

    def test_dark_mode_background(self):
        px = _img_pixel(synthesize_text_to_image("Hello", "en", dark=True))
        assert px == _CANVAS_BG_DARK

    def test_long_text_canvas_taller_than_short(self):
        _, h_short = _img_size(synthesize_text_to_image("Hi", "en"))
        long_text = (
            "This is a much longer sentence that should require additional vertical "
            "space to accommodate the wrapped text at a comfortable font size."
        )
        _, h_long = _img_size(synthesize_text_to_image(long_text, "en"))
        assert h_long >= h_short

    def test_english_output_saved(self):
        result = synthesize_text_to_image(
            "The quick brown fox jumps over the lazy dog.", "en"
        )
        out = OUTPUTS / "synth_text_en.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_english_dark_output_saved(self):
        result = synthesize_text_to_image("Dark mode translation output.", "en", dark=True)
        out = OUTPUTS / "synth_text_en_dark.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_korean_output_saved(self):
        result = synthesize_text_to_image("안녕하세요, 오늘 날씨가 좋네요.", "ko")
        out = OUTPUTS / "synth_text_ko.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_chinese_output_saved(self):
        result = synthesize_text_to_image("早上好，今天天气很好。", "zh-cn")
        out = OUTPUTS / "synth_text_zh.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_japanese_output_saved(self):
        result = synthesize_text_to_image("おはようございます。", "ja")
        out = OUTPUTS / "synth_text_ja.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# synthesize_image — validation and fake-segment tests (no EasyOCR needed)
# ---------------------------------------------------------------------------

class TestSynthesizeImage:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError):
            synthesize_image(_blank_png(), [], "hello", "en")

    def test_empty_translation_raises(self):
        with pytest.raises(ValueError):
            synthesize_image(_blank_png(), _FAKE_SEGMENTS, "", "en")

    def test_whitespace_translation_raises(self):
        with pytest.raises(ValueError):
            synthesize_image(_blank_png(), _FAKE_SEGMENTS, "   ", "en")

    def test_returns_png_bytes(self):
        result = synthesize_image(_blank_png(), _FAKE_SEGMENTS, "Good morning", "en")
        assert result[:8] == _PNG_MAGIC

    def test_output_same_dimensions_as_source(self):
        src_w, src_h = 400, 100
        result = synthesize_image(_blank_png(src_w, src_h), _FAKE_SEGMENTS, "Hello", "en")
        assert _img_size(result) == (src_w, src_h)

    def test_dark_source_output_saved(self):
        result = synthesize_image(
            _blank_png(400, 100, (30, 30, 30)), _FAKE_SEGMENTS, "Dark background test", "en"
        )
        out = OUTPUTS / "synth_image_dark_bg.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_light_source_output_saved(self):
        result = synthesize_image(
            _blank_png(400, 100, (240, 240, 240)), _FAKE_SEGMENTS, "Light background test", "en"
        )
        out = OUTPUTS / "synth_image_light_bg.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    def test_korean_translation_on_blank(self):
        result = synthesize_image(_blank_png(), _FAKE_SEGMENTS, "안녕하세요", "ko")
        out = OUTPUTS / "synth_image_ko.png"
        out.write_bytes(result)
        assert result[:8] == _PNG_MAGIC

    # -----------------------------------------------------------------------
    # Integration — requires EasyOCR + real test image asset
    # -----------------------------------------------------------------------

    @pytest.mark.integration
    def test_real_image_asset_round_trip(self):
        """OCR the real test image, then synthesize the translation back in."""
        from ocr import extract_text
        path = _asset("test_image.png")
        segments = extract_text(path)
        if not segments:
            pytest.skip("OCR returned no segments for test_image.png")
        result = synthesize_image(path, segments, "This is a synthesized translation.", "en")
        assert result[:8] == _PNG_MAGIC
        out = OUTPUTS / "synth_image_real.png"
        out.write_bytes(result)
        assert out.stat().st_size > 0

    @pytest.mark.integration
    def test_second_real_image_asset(self):
        from ocr import extract_text
        path = _asset("test_image_2.png")
        segments = extract_text(path)
        if not segments:
            pytest.skip("OCR returned no segments for test_image_2.png")
        result = synthesize_image(path, segments, "Synthesized output for image 2.", "en")
        out = OUTPUTS / "synth_image_real_2.png"
        out.write_bytes(result)
        assert result[:8] == _PNG_MAGIC
