"""Tests for translate_image.py — OCR and text translation are mocked."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1-Text"))

from translate_image import translate_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _text_result(src_lang="es", translated="hello", confidence=0.99, method="opus-mt"):
    return {
        "translated_text": translated,
        "source_language": src_lang,
        "confidence": confidence,
        "method": method,
    }


# ---------------------------------------------------------------------------
# translate_image — result is {"auto": {...}, "hint": None, "collected_path": ...}
# ---------------------------------------------------------------------------

class TestTranslateImage:
    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_auto_result_has_expected_keys(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("hola mundo", 0.95)
        mock_translate.return_value = _text_result("es", "hello world", 0.99)

        result = translate_image(_blank_image())
        auto = result["auto"]

        for key in ("original_text", "translated_text", "source_language", "confidence",
                    "ocr_confidence", "method", "score"):
            assert key in auto

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_ocr_confidence_propagated(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("早上好", 0.91)
        mock_translate.return_value = _text_result("zh-cn", "Good morning", 1.0)

        result = translate_image(_blank_image())
        assert result["auto"]["ocr_confidence"] == 0.91

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_original_text_preserved(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("buenos noches", 0.88)
        mock_translate.return_value = _text_result("es", "good evening", 0.97)

        result = translate_image(_blank_image())
        assert result["auto"]["original_text"] == "buenos noches"
        assert result["auto"]["translated_text"] == "good evening"

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.extract_text_combined")
    def test_empty_ocr_returns_none_method(self, mock_ocr, _save):
        mock_ocr.return_value = ("", 0.0)

        result = translate_image(_blank_image())
        auto = result["auto"]
        assert auto["method"] == "none"
        assert auto["original_text"] == ""
        assert auto["translated_text"] == ""
        assert auto["ocr_confidence"] == 0.0

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_src_lang_forwarded_to_translate(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("konnichiwa", 0.80)
        mock_translate.return_value = _text_result("ja", "hello", 0.75)

        translate_image(_blank_image(), src_lang="ja")
        mock_translate.assert_called_once_with("konnichiwa", src_lang="ja", tgt_lang=None)

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_english_passthrough_method(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("hello world", 0.99)
        mock_translate.return_value = _text_result("en", "hello world", 0.99, "passthrough")

        result = translate_image(_blank_image())
        assert result["auto"]["method"] == "passthrough"

    @patch("translate_image.save_submission", return_value=None)
    @patch("translate_image.translate_text")
    @patch("translate_image.extract_text_combined")
    def test_hint_none_when_no_hint_given(self, mock_ocr, mock_translate, _save):
        mock_ocr.return_value = ("hola", 0.90)
        mock_translate.return_value = _text_result()

        result = translate_image(_blank_image())
        assert result["hint"] is None
