"""Tests for translate.py — HuggingFace API calls are mocked."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from translate import translate_text, translate_to_english


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

    @patch("translate._client")
    def test_non_english_uses_opus_mt(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("My name is Wolfgang")
        result = translate_text("Меня зовут Вольфганг", src_lang="ru")
        assert result["method"] == "opus-mt"
        assert result["source_language"] == "ru"
        assert result["translated_text"] == "My name is Wolfgang"

    @patch("translate._client")
    def test_calls_translation_api(self, mock_client_fn):
        client = _mock_client("Good morning")
        mock_client_fn.return_value = client
        translate_text("早上好", src_lang="zh-cn")
        client.translation.assert_called_once_with(
            "早上好", model="Helsinki-NLP/opus-mt-mul-en"
        )

    @patch("translate._client")
    def test_result_keys_present(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("translated")
        result = translate_text("Bonjour le monde", src_lang="fr")
        assert "translated_text" in result
        assert "source_language" in result
        assert "method" in result


class TestTranslateToEnglish:
    def test_calls_client_translation(self):
        client = _mock_client("hello world")
        result = translate_to_english("hola mundo", client)
        client.translation.assert_called_once_with(
            "hola mundo", model="Helsinki-NLP/opus-mt-mul-en"
        )
        assert result == "hello world"

    def test_returns_string(self):
        client = _mock_client("good evening")
        result = translate_to_english("buenas noches", client)
        assert result == "good evening"
