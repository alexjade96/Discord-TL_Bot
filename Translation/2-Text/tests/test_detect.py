"""Tests for detect.py — no API calls required."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detect import detect_language, is_english, dominant_language
from utils import get_mbart_code


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("The quick brown fox jumps over the lazy dog") == "en"

    def test_russian(self):
        assert detect_language("Меня зовут Вольфганг и я живу в Берлине") == "ru"

    def test_chinese_simplified(self):
        assert detect_language("早上好，今天天气怎么样？") == "zh-cn"

    def test_japanese(self):
        assert detect_language("おはようございます") == "ja"

    def test_korean(self):
        assert detect_language("안녕하세요") == "ko"

    def test_empty_returns_unknown(self):
        assert detect_language("") == "unknown"

    def test_very_short_falls_back_gracefully(self):
        result = detect_language("hi")
        assert isinstance(result, str)


class TestIsEnglish:
    def test_english_sentence(self):
        assert is_english("Hello, how are you today?") is True

    def test_non_english(self):
        assert is_english("Bonjour le monde") is False

    def test_chinese(self):
        assert is_english("早上好") is False


class TestDominantLanguage:
    def test_returns_string(self):
        result = dominant_language("Hello world")
        assert isinstance(result, str)

    def test_english_dominant(self):
        assert dominant_language("This is clearly English text with many words.") == "en"

    def test_empty_returns_unknown(self):
        assert dominant_language("") == "unknown"


class TestGetMbartCode:
    def test_russian(self):
        assert get_mbart_code("ru") == "ru_RU"

    def test_chinese_simplified(self):
        assert get_mbart_code("zh-cn") == "zh_CN"

    def test_chinese_traditional(self):
        assert get_mbart_code("zh-tw") == "zh_TW"

    def test_japanese(self):
        assert get_mbart_code("ja") == "ja_XX"

    def test_korean(self):
        assert get_mbart_code("ko") == "ko_KR"

    def test_english(self):
        assert get_mbart_code("en") == "en_XX"

    def test_unsupported_returns_none(self):
        assert get_mbart_code("xx") is None

    def test_case_insensitive(self):
        assert get_mbart_code("RU") == "ru_RU"
        assert get_mbart_code("ZH-CN") == "zh_CN"
