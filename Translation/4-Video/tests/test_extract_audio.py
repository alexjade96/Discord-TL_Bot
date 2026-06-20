"""Tests for extract_audio.py — PyAV is mocked to avoid real media files."""

import io
import struct
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extract_audio import extract_audio, _write_wav, _TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# _write_wav — pure function, no mocking needed
# ---------------------------------------------------------------------------

class TestWriteWav:
    def test_riff_magic(self):
        wav = _write_wav(np.zeros(100, dtype=np.int16), 16000)
        assert wav[:4] == b"RIFF"

    def test_wave_tag(self):
        wav = _write_wav(np.zeros(100, dtype=np.int16), 16000)
        assert wav[8:12] == b"WAVE"

    def test_fmt_chunk(self):
        wav = _write_wav(np.zeros(100, dtype=np.int16), 16000)
        assert wav[12:16] == b"fmt "

    def test_sample_rate_in_header(self):
        wav = _write_wav(np.zeros(100, dtype=np.int16), 16000)
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        assert sample_rate == 16000

    def test_total_length_correct(self):
        n = 500
        wav = _write_wav(np.zeros(n, dtype=np.int16), 16000)
        # 44-byte header + 2 bytes per int16 sample
        assert len(wav) == 44 + n * 2

    def test_data_chunk_tag(self):
        wav = _write_wav(np.zeros(10, dtype=np.int16), 8000)
        assert wav[36:40] == b"data"

    def test_different_sample_rates(self):
        for rate in (8000, 22050, 44100, 48000):
            wav = _write_wav(np.zeros(10, dtype=np.int16), rate)
            sr = struct.unpack_from("<I", wav, 24)[0]
            assert sr == rate


# ---------------------------------------------------------------------------
# extract_audio — mocks av.open and av.AudioResampler
# ---------------------------------------------------------------------------

def _make_mock_container(sample_rate=24000, n_frames=2):
    """Return a mock av.InputContainer with one audio stream."""
    stream = MagicMock()
    stream.type = "audio"
    stream.sample_rate = sample_rate

    frames = []
    for _ in range(n_frames):
        frame = MagicMock()
        frames.append(frame)

    container = MagicMock()
    container.streams = [stream]
    container.decode.return_value = iter(frames)
    container.__enter__ = lambda s: s
    container.__exit__ = MagicMock(return_value=False)
    return container, stream, frames


def _make_mock_resampler(n_samples=256):
    """Return a mock AudioResampler whose resample() yields one int16 chunk."""
    resampled = MagicMock()
    resampled.to_ndarray.return_value = np.zeros((1, n_samples), dtype=np.int16)

    resampler = MagicMock()
    # First call (per frame) returns one resampled frame; flush call returns nothing
    resampler.resample.side_effect = lambda frame: [resampled] if frame is not None else []
    return resampler


class TestExtractAudio:
    @patch("extract_audio.av.AudioResampler")
    @patch("extract_audio.av.open")
    def test_returns_bytes(self, mock_open, mock_resampler_cls):
        container, _, _ = _make_mock_container()
        mock_open.return_value = container
        mock_resampler_cls.return_value = _make_mock_resampler()
        result = extract_audio(b"\x00" * 16)
        assert isinstance(result, bytes)

    @patch("extract_audio.av.AudioResampler")
    @patch("extract_audio.av.open")
    def test_output_is_valid_wav(self, mock_open, mock_resampler_cls):
        container, _, _ = _make_mock_container()
        mock_open.return_value = container
        mock_resampler_cls.return_value = _make_mock_resampler()
        result = extract_audio(b"\x00" * 16)
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    @patch("extract_audio.av.AudioResampler")
    @patch("extract_audio.av.open")
    def test_bytes_input_wrapped_in_bytesio(self, mock_open, mock_resampler_cls):
        """Bytes input should be wrapped in BytesIO before passing to av.open."""
        container, _, _ = _make_mock_container()
        mock_open.return_value = container
        mock_resampler_cls.return_value = _make_mock_resampler()
        extract_audio(b"\x00" * 16)
        open_arg = mock_open.call_args[0][0]
        assert isinstance(open_arg, io.BytesIO)

    @patch("extract_audio.av.AudioResampler")
    @patch("extract_audio.av.open")
    def test_string_path_passed_directly(self, mock_open, mock_resampler_cls):
        container, _, _ = _make_mock_container()
        mock_open.return_value = container
        mock_resampler_cls.return_value = _make_mock_resampler()
        extract_audio("/some/video.mkv")
        mock_open.assert_called_once_with("/some/video.mkv")

    @patch("extract_audio.av.open")
    def test_no_audio_stream_raises(self, mock_open):
        container = MagicMock()
        video_stream = MagicMock()
        video_stream.type = "video"
        container.streams = [video_stream]
        container.__enter__ = lambda s: s
        container.__exit__ = MagicMock(return_value=False)
        container.close = MagicMock()
        mock_open.return_value = container
        with pytest.raises(RuntimeError, match="No audio stream"):
            extract_audio(b"\x00" * 16)

    @patch("extract_audio.av.AudioResampler")
    @patch("extract_audio.av.open")
    def test_resampler_targets_16khz_mono(self, mock_open, mock_resampler_cls):
        container, _, _ = _make_mock_container()
        mock_open.return_value = container
        resampler = _make_mock_resampler()
        mock_resampler_cls.return_value = resampler
        extract_audio(b"\x00" * 16)
        mock_resampler_cls.assert_called_once_with(format="s16", layout="mono", rate=_TARGET_SAMPLE_RATE)
