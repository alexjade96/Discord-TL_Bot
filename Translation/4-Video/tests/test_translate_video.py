"""Tests for translate_video.py — extract_audio, transcribe, and translate_text are mocked."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "1-Text"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "3-Audio"))

import pytest
from translate_video import translate_video


def _transcription(text="안녕하세요", lang="ko", confidence=0.99):
    return {"transcript": text, "source_language": lang, "confidence": confidence, "method": "whisper-hf"}


def _translation(text="Hello", lang="ko", method="opus-mt"):
    return {"translated_text": text, "source_language": lang, "confidence": 0.99, "method": method}


WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20 + b"data\x00\x00\x00\x00"


class TestTranslateVideo:
    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_result_keys_present(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        result = translate_video(b"\x00" * 16)
        for key in ("original_text", "translated_text", "source_language", "confidence", "method", "collected"):
            assert key in result

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_transcript_becomes_original_text(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription("안녕하세요")
        mock_translate.return_value = _translation("Hello")
        result = translate_video(b"\x00" * 16)
        assert result["original_text"] == "안녕하세요"
        assert result["translated_text"] == "Hello"

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_extract_audio_called_with_source(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        source = b"\xAB\xCD" * 8
        translate_video(source)
        mock_extract.assert_called_once_with(source)

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_transcribe_receives_wav_bytes(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        translate_video(b"\x00" * 16)
        mock_transcribe.assert_called_once_with(WAV_BYTES, src_lang=None)

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_empty_transcript_short_circuits(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription("")
        result = translate_video(b"\x00" * 16)
        mock_translate.assert_not_called()
        assert result["original_text"] == ""
        assert result["translated_text"] == ""
        assert result["collected"] is False

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_from_lang_forwarded_to_transcribe(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        translate_video(b"\x00" * 16, from_lang="ja")
        mock_transcribe.assert_called_once_with(WAV_BYTES, src_lang="ja")

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_to_lang_forwarded_to_translate(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation(method="opus-mt")
        translate_video(b"\x00" * 16, to_lang="fr")
        call_kwargs = mock_translate.call_args[1]
        assert call_kwargs.get("tgt_lang") == "fr"

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_method_comes_from_transcription(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        result = translate_video(b"\x00" * 16)
        assert result["method"] == "whisper-hf"

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_passthrough_not_collected(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription("hello", lang="en")
        mock_translate.return_value = _translation("hello", lang="en", method="passthrough")
        result = translate_video(b"\x00" * 16)
        assert result["collected"] is False

    @patch("translate_video.translate_text")
    @patch("translate_video.transcribe")
    @patch("translate_video.extract_audio")
    def test_path_source_accepted(self, mock_extract, mock_transcribe, mock_translate):
        mock_extract.return_value = WAV_BYTES
        mock_transcribe.return_value = _transcription()
        mock_translate.return_value = _translation()
        p = Path("video.mkv")
        translate_video(p)
        mock_extract.assert_called_once_with(p)
