"""Tests for translate.py — OCR and text translation are mocked."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "2-Text"))

from translate import translate_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _text_result(src_lang="es", translated="hello", confidence=0.99):
    return {
        "translated_text": translated,
        "source_language": src_lang,
        "confidence": confidence,
        "method": "opus-mt",
    }


# ---------------------------------------------------------------------------
# translate_image
# ---------------------------------------------------------------------------

class TestTranslateImage:
    @patch("translate.translate_text")
    @patch("translate.extract_text_combined")
    def test_full_pipeline_returns_expected_keys(self, mock_ocr, mock_translate):
        mock_ocr.return_value = ("hola mundo", 0.95)
        mock_translate.return_value = _text_result("es", "hello world", 0.99)

        result = translate_image(_blank_image())

        assert "original_text" in result
        assert "translated_text" in result
        assert "source_language" in result
        assert "confidence" in result
        assert "ocr_confidence" in result
        assert "method" in result

    @patch("translate.translate_text")
    @patch("translate.extract_text_combined")
    def test_ocr_confidence_propagated(self, mock_ocr, mock_translate):
        mock_ocr.return_value = ("早上好", 0.91)
        mock_translate.return_value = _text_result("zh-cn", "Good morning", 1.0)

        result = translate_image(_blank_image())
        assert result["ocr_confidence"] == 0.91

    @patch("translate.translate_text")
    @patch("translate.extract_text_combined")
    def test_original_text_preserved(self, mock_ocr, mock_translate):
        mock_ocr.return_value = ("buenos noches", 0.88)
        mock_translate.return_value = _text_result("es", "good evening", 0.97)

        result = translate_image(_blank_image())
        assert result["original_text"] == "buenos noches"
        assert result["translated_text"] == "good evening"

    @patch("translate.extract_text_combined")
    def test_empty_ocr_returns_none_method(self, mock_ocr):
        mock_ocr.return_value = ("", 0.0)

        result = translate_image(_blank_image())
        assert result["method"] == "none"
        assert result["original_text"] == ""
        assert result["translated_text"] == ""
        assert result["ocr_confidence"] == 0.0

    @patch("translate.translate_text")
    @patch("translate.extract_text_combined")
    def test_src_lang_forwarded_to_translate(self, mock_ocr, mock_translate):
        mock_ocr.return_value = ("konnichiwa", 0.80)
        mock_translate.return_value = _text_result("ja", "hello", 0.75)

        translate_image(_blank_image(), src_lang="ja")
        mock_translate.assert_called_once_with("konnichiwa", src_lang="ja")

    @patch("translate.translate_text")
    @patch("translate.extract_text_combined")
    def test_english_passthrough_method(self, mock_ocr, mock_translate):
        mock_ocr.return_value = ("hello world", 0.99)
        mock_translate.return_value = {
            "translated_text": "hello world",
            "source_language": "en",
            "confidence": 0.99,
            "method": "passthrough",
        }

        result = translate_image(_blank_image())
        assert result["method"] == "passthrough"
