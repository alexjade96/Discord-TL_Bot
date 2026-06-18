"""Tests for transcribe_audio.py — HuggingFace API calls are mocked."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# transcribe_audio.py imports detect.py from 1-Text; ensure it's on the path first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "1-Text"))

import pytest
from transcribe_audio import transcribe


def _mock_asr(text="안녕하세요"):
    result = MagicMock()
    result.text = text
    return result


class TestTranscribe:
    @patch("transcribe_audio._client")
    def test_result_keys_present(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("hello")
        result = transcribe("fake.ogg")
        for key in ("transcript", "source_language", "confidence", "method"):
            assert key in result

    @patch("transcribe_audio._client")
    def test_method_is_whisper_hf(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("hello")
        result = transcribe("fake.ogg")
        assert result["method"] == "whisper-hf"

    @patch("transcribe_audio._client")
    def test_transcript_returned(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("안녕하세요")
        result = transcribe("fake.ogg")
        assert result["transcript"] == "안녕하세요"

    @patch("transcribe_audio._client")
    def test_empty_transcript_returns_unknown_language(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("")
        result = transcribe("fake.ogg")
        assert result["transcript"] == ""
        assert result["source_language"] == "unknown"
        assert result["confidence"] is None

    @patch("transcribe_audio._client")
    def test_src_lang_hint_bypasses_detection(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("hello")
        result = transcribe("fake.ogg", src_lang="ko")
        assert result["source_language"] == "ko"
        assert result["confidence"] is None

    @patch("transcribe_audio._client")
    def test_bytes_input_calls_api_with_path(self, mock_client_fn):
        """Bytes input must be written to a temp file — the API call receives a path, not bytes."""
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("test")
        wav_bytes = b"RIFF" + b"\x00" * 100
        result = transcribe(wav_bytes)
        assert result["transcript"] == "test"
        call_args = mock_client_fn.return_value.automatic_speech_recognition.call_args
        assert isinstance(call_args[0][0], str)

    @patch("transcribe_audio._client")
    def test_path_input_passed_as_string(self, mock_client_fn):
        mock_client_fn.return_value.automatic_speech_recognition.return_value = _mock_asr("hi")
        transcribe(Path("some_file.ogg"))
        call_args = mock_client_fn.return_value.automatic_speech_recognition.call_args
        assert isinstance(call_args[0][0], str)
