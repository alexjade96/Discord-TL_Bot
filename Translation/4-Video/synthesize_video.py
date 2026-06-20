"""Video re-synthesis: replace a video's audio track with synthesized translated speech.

Takes the original video and translated text, synthesizes new audio via gTTS, and
remuxes the original video stream(s) with the new audio into an MKV container.
PyAV handles all container I/O — no system ffmpeg binary required.

This is the output side of the video pipeline:

    source video → extract audio → transcribe → translate → synthesize_video → Discord file

Usage (CLI):
    python synthesize_video.py video.mp4 "안녕하세요" --lang ko
    python synthesize_video.py clip.mkv "Hello world" --lang en --out out.mkv
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "3-Audio"))

import av

from synthesize_audio import synthesize as synthesize_speech  # noqa: E402

# Opus encoder sample rate; 48 kHz is the native rate for the libopus codec.
_OPUS_SAMPLE_RATE = 48_000


def synthesize_video(
    source: bytes | str | Path,
    translated_text: str,
    tgt_lang: str,
) -> bytes:
    """Remux a video with its audio track replaced by synthesized translated speech.

    The original video stream(s) are copied packet-for-packet (no re-encode).
    The new audio is synthesized via gTTS and encoded as Opus in the output MKV.

    If the source video has no video stream (audio-only container), the output
    will contain only the new synthesized audio track.

    Args:
        source:          Video as raw bytes, a local file path (str or Path), or URL.
        translated_text: The translated transcript to speak.
        tgt_lang:        Language code for speech synthesis (e.g. 'en', 'ko', 'zh-CN').

    Returns:
        MKV bytes with video stream(s) from the original and new synthesized audio.

    Raises:
        ValueError:      If translated_text is empty.
        RuntimeError:    If the source container cannot be opened.
        gtts.gTTSError:  If the gTTS request fails.
    """
    if not translated_text or not translated_text.strip():
        raise ValueError("Cannot synthesize video with empty translated text.")

    audio_mp3 = synthesize_speech(translated_text, tgt_lang)

    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as tmp:
        out_path = tmp.name

    try:
        in_video = _open_source(source)
        in_audio = av.open(io.BytesIO(audio_mp3))
        try:
            return _remux(in_video, in_audio, out_path)
        finally:
            in_video.close()
            in_audio.close()
    finally:
        Path(out_path).unlink(missing_ok=True)


def _open_source(source: bytes | str | Path) -> av.container.InputContainer:
    if isinstance(source, bytes):
        return av.open(io.BytesIO(source))
    return av.open(str(source))


def _remux(
    in_video: av.container.InputContainer,
    in_audio: av.container.InputContainer,
    out_path: str,
) -> bytes:
    """Copy video streams from in_video and replace audio with in_audio → MKV bytes."""
    video_streams = [s for s in in_video.streams if s.type == "video"]

    with av.open(out_path, mode="w", format="matroska") as out_c:
        # Map each original video stream to an output stream (packet-level copy).
        stream_map: dict[int, av.stream.Stream] = {}
        for vs in video_streams:
            out_stream = out_c.add_stream(template=vs)
            stream_map[vs.index] = out_stream

        # New audio stream: synthesized speech encoded as Opus at 48 kHz.
        out_audio = out_c.add_stream("libopus", rate=_OPUS_SAMPLE_RATE)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=_OPUS_SAMPLE_RATE)

        # Remux video packets without decoding.
        if video_streams:
            for packet in in_video.demux(*video_streams):
                if packet.dts is None:
                    continue
                packet.stream = stream_map[packet.stream_index]
                out_c.mux(packet)

        # Decode, resample, and encode the synthesized audio.
        audio_stream = next(s for s in in_audio.streams if s.type == "audio")
        for frame in in_audio.decode(audio_stream):
            for resampled in resampler.resample(frame):
                resampled.pts = None
                for packet in out_audio.encode(resampled):
                    out_c.mux(packet)
        for resampled in resampler.resample(None):
            resampled.pts = None
            for packet in out_audio.encode(resampled):
                out_c.mux(packet)
        for packet in out_audio.encode(None):
            out_c.mux(packet)

    return Path(out_path).read_bytes()


if __name__ == "__main__":
    import argparse
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Replace a video's audio track with synthesized translated speech."
    )
    parser.add_argument("source", help="Video file path or URL")
    parser.add_argument("text", help="Translated text to synthesize as new audio")
    parser.add_argument("--lang", default="en", help="Speech language code (default: en)")
    parser.add_argument("--out", default=None, help="Output MKV path (default: output.mkv)")
    args = parser.parse_args()

    result = synthesize_video(args.source, args.text, args.lang)
    out_path = Path(args.out) if args.out else Path("output.mkv")
    out_path.write_bytes(result)
    print(f"Wrote {len(result):,} bytes → {out_path}")
