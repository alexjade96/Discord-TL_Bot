"""Tests for translate_audio.py — transcribe and translate_text are mocked."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "1-Text"))

import pytest
from translate_audio import translate_audio


def _transcription(text="안녕하세요", lang="ko", confidence=0.99):
    return {"transcript": text, "source_language": lang, "confidence": confidence, "method": "whisper-hf"}


def _translation(text="Hello", lang="ko", method="opus-mt"):
    return {"translated_text": text, "source_language": lang, "confidence": 0.99, "method": method}


class TestTranslateAudio:
    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_result_keys_present(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        result = translate_audio("fake.ogg")
        for key in ("original_text", "translated_text", "source_language", "confidence", "method", "collected"):
            assert key in result

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_transcript_becomes_original_text(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription("안녕하세요")
        mock_translate.return_value = _translation("Hello")
        result = translate_audio("fake.ogg")
        assert result["original_text"] == "안녕하세요"
        assert result["translated_text"] == "Hello"

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_empty_transcript_short_circuits(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription("")
        result = translate_audio("fake.ogg")
        mock_translate.assert_not_called()
        assert result["original_text"] == ""
        assert result["translated_text"] == ""
        assert result["collected"] is False

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_from_lang_forwarded_to_transcribe(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        translate_audio("fake.ogg", from_lang="ko")
        mock_transcribe.assert_called_once_with("fake.ogg", src_lang="ko")

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_to_lang_forwarded_to_translate(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation(method="opus-mt")
        translate_audio("fake.ogg", to_lang="fr")
        call_kwargs = mock_translate.call_args[1]
        assert call_kwargs.get("tgt_lang") == "fr"

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_method_comes_from_transcription(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        result = translate_audio("fake.ogg")
        assert result["method"] == "whisper-hf"

    @patch("translate_audio.translate_text")
    @patch("translate_audio.transcribe")
    def test_passthrough_not_collected(self, mock_transcribe, mock_translate):
        mock_transcribe.return_value = _transcription("hello", lang="en")
        mock_translate.return_value = _translation("hello", lang="en", method="passthrough")
        result = translate_audio("fake.ogg")
        assert result["collected"] is False
