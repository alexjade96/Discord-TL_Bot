"""Tests for translate.py — HuggingFace API calls are mocked."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from translate import translate_text, translate_with_qwen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(generation_text="generated output"):
    client = MagicMock()
    client.text_generation.return_value = generation_text
    return client


# ---------------------------------------------------------------------------
# translate_text
# ---------------------------------------------------------------------------

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
    def test_russian_uses_qwen(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("My name is Wolfgang")
        result = translate_text("Меня зовут Вольфганг", src_lang="ru")
        assert result["method"] == "qwen"
        assert result["source_language"] == "ru"
        assert result["translated_text"] == "My name is Wolfgang"

    @patch("translate._client")
    def test_non_english_uses_qwen(self, mock_client_fn):
        client = _mock_client("Good morning")
        mock_client_fn.return_value = client
        result = translate_text("早上好", src_lang="zh-cn")
        assert result["method"] == "qwen"
        client.text_generation.assert_called_once()

    @patch("translate._client")
    def test_result_keys_present(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("translated")
        result = translate_text("Bonjour le monde", src_lang="fr")
        assert "translated_text" in result
        assert "source_language" in result
        assert "method" in result


# ---------------------------------------------------------------------------
# translate_with_qwen (unit)
# ---------------------------------------------------------------------------

class TestTranslateWithQwen:
    def test_calls_text_generation(self):
        client = _mock_client(generation_text="  translated  ")
        result = translate_with_qwen("hola", client)
        client.text_generation.assert_called_once()
        assert result == "translated"  # leading/trailing whitespace stripped

    def test_prompt_contains_input_text(self):
        client = _mock_client(generation_text="out")
        translate_with_qwen("unique_input_string_xyz", client)
        prompt_arg = client.text_generation.call_args[0][0]
        assert "unique_input_string_xyz" in prompt_arg
