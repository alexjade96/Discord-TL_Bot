"""Tests for translate_text.py — HuggingFace API calls are mocked."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from translate_text import translate_text


def _mock_client(translation_text="translated output"):
    client = MagicMock()
    client.translation.return_value = MagicMock(translation_text=translation_text)
    return client


class TestTranslateText:
    def test_empty_string_returns_none_method(self):
        result = translate_text("")
        assert result["method"] == "none"
        assert result["translated_text"] == ""

    def test_whitespace_only_returns_none_method(self):
        result = translate_text("   ")
        assert result["method"] == "none"

    def test_english_passthrough(self):
        result = translate_text("Hello, how are you?")
        assert result["method"] == "passthrough"
        assert result["source_language"] == "en"
        assert result["translated_text"] == "Hello, how are you?"

    def test_explicit_src_en_passthrough(self):
        result = translate_text("Bonjour", src_lang="en")
        assert result["method"] == "passthrough"

    @patch("translate_text._client")
    def test_non_english_uses_opus_mt(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("My name is Wolfgang")
        result = translate_text("Меня зовут Вольфганг", src_lang="ru")
        assert result["method"] == "opus-mt"
        assert result["source_language"] == "ru"
        assert result["translated_text"] == "My name is Wolfgang"

    @patch("translate_text._client")
    def test_calls_translation_api_with_src_lang(self, mock_client_fn):
        client = _mock_client("Good morning")
        mock_client_fn.return_value = client
        translate_text("早上好", src_lang="zh-cn")
        client.translation.assert_called_once_with(
            "早上好", model="Helsinki-NLP/opus-mt-mul-en", src_lang="zh", tgt_lang="en"
        )

    @patch("translate_text._client")
    def test_result_keys_present(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("translated")
        result = translate_text("Bonjour le monde", src_lang="fr")
        assert "translated_text" in result
        assert "source_language" in result
        assert "method" in result

    @patch("translate_text._client")
    def test_same_src_as_tgt_is_passthrough(self, mock_client_fn):
        """Source and target resolving to the same language returns passthrough."""
        result = translate_text("안녕하세요", src_lang="ko", tgt_lang="ko")
        mock_client_fn.assert_not_called()
        assert result["method"] == "passthrough"
