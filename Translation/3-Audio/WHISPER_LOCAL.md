# Local Whisper Implementation Guide

This file documents how to migrate the audio pipeline from HuggingFace Inference
API to a locally-hosted Whisper model, following the same local-first pattern
already used in `translate_text.py`.

The current implementation (`transcribe_audio.py`) calls `openai/whisper-large-v3`
via the HF API. Once enough corrected submissions exist in
`Translation/0-Data/Audio/data/audio_submissions.jsonl`, fine-tuning on your
specific domain (Discord voice messages, compressed Opus audio, code-switched
speech) will measurably outperform the base model.

---

## When to Switch

Switch when you have collected enough audio submissions with manual `correct_transcript`
corrections to make fine-tuning worthwhile. Review coverage via:

```bash
.venv\Scripts\python.exe Translation\0-Data\Audio\training\collect_audio.py
```

As a rough guide: 100+ corrected entries per dominant language before fine-tuning.

---

## Dependencies

```bash
pip install openai-whisper
# or, if using transformers (same package already installed):
# transformers already supports WhisperForConditionalGeneration
pip install librosa soundfile  # audio loading utilities
```

---

## Step 1 — Update `transcribe_audio.py`

Add local model support following the same pattern as `translate_text.py`:

```python
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

_LOCAL_WHISPER_DIR = Path.home() / ".tl-bot" / "whisper"
_whisper_cache: dict[str, tuple] = {}

def _load_local_whisper(model_size: str = "large-v3"):
    """Return (model, processor) for a locally deployed Whisper model, or None."""
    key = model_size
    if key in _whisper_cache:
        return _whisper_cache[key]
    model_dir = _LOCAL_WHISPER_DIR / model_size
    if not model_dir.exists():
        _whisper_cache[key] = None
        return None
    try:
        processor = WhisperProcessor.from_pretrained(str(model_dir))
        model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
        model.eval()
        _whisper_cache[key] = (model, processor)
        return _whisper_cache[key]
    except Exception:
        _whisper_cache[key] = None
        return None

def _run_local_whisper(audio_bytes: bytes, model, processor, src_lang=None) -> str:
    import torch, librosa, io
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    forced_ids = processor.get_decoder_prompt_ids(language=src_lang, task="transcribe") if src_lang else None
    with torch.no_grad():
        generated = model.generate(**inputs, forced_decoder_ids=forced_ids)
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
```

Then update `transcribe()` to check for local model first:

```python
def transcribe(source, src_lang=None):
    # Resolve to bytes if URL/path
    if isinstance(source, str) and source.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(source) as r:
            raw = r.read()
    elif isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source  # already bytes

    # 1. Local fine-tuned model
    local = _load_local_whisper()
    if local:
        transcript = _run_local_whisper(raw, *local, src_lang=src_lang)
        method = "whisper-local"
    else:
        # 2. HF Inference API fallback
        result = _client().automatic_speech_recognition(raw, model=WHISPER_MODEL)
        transcript = (result.text or "").strip()
        method = "whisper-hf"

    # Language detection unchanged
    ...
    return {"transcript": transcript, ..., "method": method}
```

---

## Step 2 — Build the Training Dataset

Add `dataset.py` to `Translation/0-Data/Audio/training/` mirroring the text
pipeline. Audio training pairs are `(audio_file, correct_transcript)` rather
than `(source_text, target_text)`.

Whisper fine-tuning expects:
- `input_features`: mel spectrogram from `WhisperProcessor`
- `labels`: tokenized transcript (prefer `correct_transcript` over `transcript`)

Use `correct_transcript` when set, fall back to `transcript`:

```python
def _best_transcript(entry: dict) -> str:
    return entry.get("correct_transcript") or entry.get("transcript") or ""
```

The audio files themselves need to be re-retrieved or stored locally at collection
time. Update `collect_audio.save_submission()` to optionally save audio bytes to
`data/audio/` alongside the JSONL if local training is planned.

---

## Step 3 — Fine-Tune

Use `WhisperForConditionalGeneration` with `Seq2SeqTrainer` (same trainer class
as the text `train.py`). Key differences from text fine-tuning:

- Input: `input_features` (mel spectrogram) instead of tokenized text
- Loss: cross-entropy over decoder token sequence (same as MarianMT)
- Batch size: smaller (4–8) due to audio memory overhead
- Learning rate: 1e-5 (Whisper is sensitive to higher rates)

Official HuggingFace guide:
https://huggingface.co/blog/fine-tune-whisper

---

## Step 4 — Deploy

```bash
# Copy fine-tuned checkpoint to local inference path
python deploy.py --size large-v3

# deploy.py (create alongside train.py):
#   shutil.copytree(checkpoint_dir, Path.home() / ".tl-bot" / "whisper" / "large-v3")
```

`transcribe_audio.py` will detect and use the local model automatically on next
load — no bot restart required beyond module cache clearing.

---

## Corrections Schema Reminder

`audio_submissions.jsonl` already has two correction fields:

| Field | Purpose |
|---|---|
| `correct_transcript` | Human-corrected Whisper output — primary ASR training signal |
| `correct_translation` | Human-corrected translation of the transcript |

Use `review.py --mode audio` once audio mode is added to the review CLI.
