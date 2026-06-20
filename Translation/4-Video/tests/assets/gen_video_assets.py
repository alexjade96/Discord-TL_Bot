"""Generate test video assets for the 4-Video integration tests.

Run from repo root:
    python Translation/4-Video/tests/assets/gen_video_assets.py

Produces an MKV file with Korean Opus audio synthesized via gTTS and PyAV.
Requires internet access (gTTS). No HF_TOKEN needed.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import av
from synthesize_video import _OPUS_SAMPLE_RATE  # noqa: E402


def synthesize_mp3(text: str, lang: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def wrap_in_mkv(audio_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as vf:
        out_path = vf.name
    try:
        with av.open(io.BytesIO(audio_bytes)) as in_c:
            in_stream = next(s for s in in_c.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=_OPUS_SAMPLE_RATE)
            with av.open(out_path, mode="w", format="matroska") as out_c:
                out_stream = out_c.add_stream("libopus", rate=_OPUS_SAMPLE_RATE)
                for frame in in_c.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        resampled.pts = None
                        for packet in out_stream.encode(resampled):
                            out_c.mux(packet)
                for resampled in resampler.resample(None):
                    resampled.pts = None
                    for packet in out_stream.encode(resampled):
                        out_c.mux(packet)
                for packet in out_stream.encode(None):
                    out_c.mux(packet)
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


def main() -> None:
    samples = [
        ("korean",   "ko",    "안녕하세요, 오늘 날씨가 좋네요"),
        ("chinese",  "zh-CN", "早上好，今天天气很好"),
    ]
    for label, lang, text in samples:
        mp3 = synthesize_mp3(text, lang)
        mkv = wrap_in_mkv(mp3)
        out = HERE / f"test_video_{label}.mkv"
        out.write_bytes(mkv)
        print(f"Wrote {out.name}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
