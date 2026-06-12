"""Tests for translate.py — HuggingFace API calls are mocked."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from translate import translate_text, translate_with_mbart, translate_with_qwen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(translation_text="translated output", generation_text="generated output"):
    client = MagicMock()
    client.translation.return_value = MagicMock(translation_text=translation_text)
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
    def test_russian_uses_mbart(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client("My name is Wolfgang")
        result = translate_text("Меня зовут Вольфганг", src_lang="ru")
        assert result["method"] == "mbart"
        assert result["source_language"] == "ru"
        assert result["translated_text"] == "My name is Wolfgang"

    @patch("translate._client")
    def test_mbart_receives_correct_src_code(self, mock_client_fn):
        client = _mock_client("Good morning")
        mock_client_fn.return_value = client
        translate_text("早上好", src_lang="zh-cn")
        client.translation.assert_called_once()
        call_kwargs = client.translation.call_args
        assert call_kwargs.kwargs.get("src_lang") == "zh_CN"
        assert call_kwargs.kwargs.get("tgt_lang") == "en_XX"

    @patch("translate._client")
    def test_unsupported_language_falls_back_to_qwen(self, mock_client_fn):
        client = _mock_client(generation_text="some translation")
        mock_client_fn.return_value = client
        result = translate_text("some text", src_lang="xx")
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
# translate_with_mbart (unit)
# ---------------------------------------------------------------------------

class TestTranslateWithMbart:
    def test_calls_client_translation(self):
        client = _mock_client("Berlin")
        result = translate_with_mbart("Berlin", "de_DE", client)
        client.translation.assert_called_once_with(
            "Berlin",
            model="facebook/mbart-large-50-many-to-many-mmt",
            src_lang="de_DE",
            tgt_lang="en_XX",
        )
        assert result == "Berlin"

    def test_returns_translation_text_string(self):
        client = _mock_client("hello world")
        result = translate_with_mbart("hola mundo", "es_XX", client)
        assert result == "hello world"


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
