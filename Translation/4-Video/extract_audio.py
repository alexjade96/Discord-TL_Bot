"""Extract the audio track from a video file using PyAV.

Produces a 16 kHz mono WAV in memory — the format Whisper expects.
PyAV bundles its own FFmpeg libraries, so no system ffmpeg install is needed.
Accepts video bytes, a local file path, or a URL.

Usage (CLI):
    python extract_audio.py video.mp4
    python extract_audio.py video.mp4 --out audio.wav
"""

from __future__ import annotations

import io
import struct
import tempfile
import urllib.request
from pathlib import Path

import av
import numpy as np

_TARGET_SAMPLE_RATE = 16_000
_TARGET_CHANNELS = 1


def _open_container(source: str | bytes) -> av.container.InputContainer:
    """Open an av.InputContainer from a file path or raw bytes."""
    if isinstance(source, bytes):
        return av.open(io.BytesIO(source))
    return av.open(source)


def _write_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Pack a (samples,) int16 numpy array into a WAV byte string."""
    num_samples = len(pcm)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1,                  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", data_size,
    )
    return header + pcm.tobytes()


def _decode_to_wav(container: av.container.InputContainer) -> bytes:
    """Decode the first audio stream, resample to 16 kHz mono, return WAV bytes."""
    audio_stream = next(
        (s for s in container.streams if s.type == "audio"), None
    )
    if audio_stream is None:
        raise RuntimeError("No audio stream found in the video file.")

    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=_TARGET_SAMPLE_RATE,
    )

    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()          # shape: (channels, samples)
            chunks.append(arr[0].astype(np.int16))

    # Flush resampler
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        chunks.append(arr[0].astype(np.int16))

    if not chunks:
        raise RuntimeError("Audio stream contained no decodable frames.")

    pcm = np.concatenate(chunks)
    return _write_wav(pcm, _TARGET_SAMPLE_RATE)


def extract_audio(source: bytes | str | Path) -> bytes:
    """Extract audio from a video and return 16 kHz mono WAV bytes.

    Args:
        source: Video as raw bytes, a local file path (str or Path), or a URL string.

    Returns:
        WAV audio bytes ready for Whisper transcription.

    Raises:
        RuntimeError: If no audio stream is found or decoding fails.
        urllib.error.URLError: If the URL cannot be fetched.
    """
    if isinstance(source, bytes):
        container = _open_container(source)
        try:
            return _decode_to_wav(container)
        finally:
            container.close()

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(source, tmp_path)
            container = av.open(tmp_path)
            try:
                return _decode_to_wav(container)
            finally:
                container.close()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    container = av.open(str(source))
    try:
        return _decode_to_wav(container)
    finally:
        container.close()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract audio track from a video file.")
    parser.add_argument("source", help="Video file path or URL")
    parser.add_argument("--out", default=None, help="Output WAV path (default: print byte count)")
    args = parser.parse_args()

    audio = extract_audio(args.source)
    if args.out:
        Path(args.out).write_bytes(audio)
        print(f"Wrote {len(audio):,} bytes → {args.out}")
    else:
        print(f"Extracted {len(audio):,} bytes of WAV audio.")
