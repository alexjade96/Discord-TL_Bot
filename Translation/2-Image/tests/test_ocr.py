"""Tests for ocr.py — EasyOCR calls are mocked to avoid loading ML models."""

import sys
import os
from contextlib import contextmanager
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


def _patch_readers(results):
    """Patch all three CJK readers to return the same `results`."""
    reader = _mock_reader(results)
    return (
        patch("ocr._get_reader_zh", return_value=reader),
        patch("ocr._get_reader_ja", return_value=reader),
        patch("ocr._get_reader_ko", return_value=reader),
    )


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
    def test_returns_sorted_segments(self):
        results = [
            ([[10, 80], [90, 80], [90, 100], [10, 100]], "Second line", 0.90),
            ([[10, 20], [90, 20], [90, 40], [10, 40]], "First line", 0.95),
        ]
        p1, p2, p3 = _patch_readers(results)
        with p1, p2, p3:
            segments = ocr.extract_text(_blank_image())
        assert segments[0]["text"] == "First line"
        assert segments[1]["text"] == "Second line"

    def test_segment_keys(self):
        p1, p2, p3 = _patch_readers(SAMPLE_RESULTS)
        with p1, p2, p3:
            segments = ocr.extract_text(_blank_image())
        assert all("text" in s and "confidence" in s and "bbox" in s for s in segments)

    def test_empty_text_segments_excluded(self):
        results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.99),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "Real text", 0.85),
        ]
        p1, p2, p3 = _patch_readers(results)
        with p1, p2, p3:
            segments = ocr.extract_text(_blank_image())
        assert len(segments) == 1
        assert segments[0]["text"] == "Real text"

    def test_no_sort_key_in_output(self):
        p1, p2, p3 = _patch_readers(SAMPLE_RESULTS)
        with p1, p2, p3:
            segments = ocr.extract_text(_blank_image())
        for s in segments:
            assert "_top_y" not in s

    def test_empty_image_returns_empty_list(self):
        p1, p2, p3 = _patch_readers([])
        with p1, p2, p3:
            assert ocr.extract_text(_blank_image()) == []


# ---------------------------------------------------------------------------
# extract_text_combined
# ---------------------------------------------------------------------------

class TestExtractTextCombined:
    def test_combines_text_with_space(self):
        p1, p2, p3 = _patch_readers(SAMPLE_RESULTS)
        with p1, p2, p3:
            text, _ = ocr.extract_text_combined(_blank_image())
        assert "Hello world" in text
        assert "早上好" in text

    def test_average_confidence(self):
        p1, p2, p3 = _patch_readers(SAMPLE_RESULTS)
        with p1, p2, p3:
            _, conf = ocr.extract_text_combined(_blank_image())
        expected = round((0.98 + 0.91) / 2, 4)
        assert conf == expected

    def test_no_text_returns_empty_and_zero(self):
        p1, p2, p3 = _patch_readers([])
        with p1, p2, p3:
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
