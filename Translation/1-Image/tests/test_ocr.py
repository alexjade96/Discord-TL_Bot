"""Tests for ocr.py — EasyOCR calls are mocked to avoid loading ML models."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ocr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image(h=100, w=200, channels=3) -> np.ndarray:
    return np.zeros((h, w, channels), dtype=np.uint8)


def _mock_reader(results):
    """Return a mock EasyOCR reader whose readtext() yields `results`."""
    reader = MagicMock()
    reader.readtext.return_value = results
    return reader


SAMPLE_RESULTS = [
    ([[10, 50], [100, 50], [100, 70], [10, 70]], "Hello world", 0.98),
    ([[10, 80], [90, 80], [90, 100], [10, 100]], "早上好", 0.91),
]


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_is_grayscale(self):
        img = _blank_image(50, 100, 3)
        result = ocr.preprocess(img)
        assert result.ndim == 2

    def test_output_is_doubled_in_size(self):
        img = _blank_image(50, 100, 3)
        result = ocr.preprocess(img)
        assert result.shape == (100, 200)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    @patch("ocr._get_reader")
    def test_returns_sorted_segments(self, mock_get_reader):
        # Results returned out of order (lower y first in our list but bbox higher)
        results = [
            ([[10, 80], [90, 80], [90, 100], [10, 100]], "Second line", 0.90),
            ([[10, 20], [90, 20], [90, 40], [10, 40]], "First line", 0.95),
        ]
        mock_get_reader.return_value = _mock_reader(results)
        segments = ocr.extract_text(_blank_image())
        assert segments[0]["text"] == "First line"
        assert segments[1]["text"] == "Second line"

    @patch("ocr._get_reader")
    def test_segment_keys(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader(SAMPLE_RESULTS)
        segments = ocr.extract_text(_blank_image())
        assert all("text" in s and "confidence" in s and "bbox" in s for s in segments)

    @patch("ocr._get_reader")
    def test_empty_text_segments_excluded(self, mock_get_reader):
        results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.99),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "Real text", 0.85),
        ]
        mock_get_reader.return_value = _mock_reader(results)
        segments = ocr.extract_text(_blank_image())
        assert len(segments) == 1
        assert segments[0]["text"] == "Real text"

    @patch("ocr._get_reader")
    def test_no_sort_key_in_output(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader(SAMPLE_RESULTS)
        segments = ocr.extract_text(_blank_image())
        for s in segments:
            assert "_top_y" not in s

    @patch("ocr._get_reader")
    def test_empty_image_returns_empty_list(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader([])
        assert ocr.extract_text(_blank_image()) == []


# ---------------------------------------------------------------------------
# extract_text_combined
# ---------------------------------------------------------------------------

class TestExtractTextCombined:
    @patch("ocr._get_reader")
    def test_combines_text_with_space(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader(SAMPLE_RESULTS)
        text, _ = ocr.extract_text_combined(_blank_image())
        assert "Hello world" in text
        assert "早上好" in text

    @patch("ocr._get_reader")
    def test_average_confidence(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader(SAMPLE_RESULTS)
        _, conf = ocr.extract_text_combined(_blank_image())
        expected = round((0.98 + 0.91) / 2, 4)
        assert conf == expected

    @patch("ocr._get_reader")
    def test_no_text_returns_empty_and_zero(self, mock_get_reader):
        mock_get_reader.return_value = _mock_reader([])
        text, conf = ocr.extract_text_combined(_blank_image())
        assert text == ""
        assert conf == 0.0


# ---------------------------------------------------------------------------
# load_image_from_path
# ---------------------------------------------------------------------------

class TestLoadImageFromPath:
    def test_loads_existing_test_image(self):
        image_dir = Path(__file__).parent.parent
        test_img = image_dir / "test_image.png"
        if not test_img.exists():
            pytest.skip("test_image.png not present")
        img = ocr.load_image_from_path(test_img)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ocr.load_image_from_path("nonexistent_file.png")
