# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discord bot (`TL-Bot.py`) that translates text and images for users. The bot is triggered by @mention and routes `/translate` commands based on attachment type (image, audio, text) or inline text. Image, text file, and audio translation pipelines are fully wired into the bot; video remains a stub.

## Running the Bot

```bash
python TL-Bot.py
```

Requires a `.env` file (gitignored) with:
```
DISCORD_BOT_TOKEN=your_token_here
HF_TOKEN=your_huggingface_token_here
```

Logs are written to `./logs/TL_Bot_<date>.log` (rotating, 32 MiB max, 5 backups). The `logs/` directory is auto-created at startup.

## Bot Token

The bot token **must not be hardcoded** in `TL-Bot.py`. It is loaded via `python-dotenv` from `DISCORD_BOT_TOKEN` in `.env` or the environment. Raising `RuntimeError` on startup if unset is intentional.

## Bot Commands

The bot only responds when `@TL-Bot` is the first token. Command routing:

- No args → greeting
- `/help` → command list (shows `--from`, `--to`, `--analyze` flags)
- `/translate [flags] <text>` → translate inline text
- `/translate [flags]` + image attachment → OCR then translate
- `/translate [flags]` + audio attachment → transcribe (Whisper) then translate
- `/translate [flags]` + text file attachment → read, decode, then translate
- `/translate` + video attachment → stub (not yet implemented)
- Unrecognized command → fallback message

Flags parsed by `_parse_translate_flags()` in `TL-Bot.py` (called once per `/translate` command, shared by attachment and inline paths):
- `--from <language>` — hint the source language; triggers a dual-pass comparison for images and text files
- `--to <language>` — set output language (default: English)
- `--analyze` — show segment analysis alongside the translation

Language names/codes are resolved via `parse_language_hint()` in `Translation/1-Text/utils.py` (supports common names, aliases, ISO 639-1 codes).

`_same_lang_msg(from_lang, extra="")` — shared helper in `TL-Bot.py` that formats the "nothing to translate" Discord message when source and target resolve to the same language.

### Bot Handler Structure

`on_message` is a flat router that parses flags once and dispatches to one of four self-contained async handlers — each structured as verify → perform → respond:

- `_handle_image(attachment, ...)` — size gate + magic byte check + OCR/translate + collect
- `_handle_audio(attachment, ...)` — size gate + magic byte check + Whisper transcribe + translate + collect
- `_handle_text_file(attachment, ...)` — size gate + safety check + decode + truncate + translate + collect
- `_handle_text_inline(text, ...)` — same-lang check + `_run_text_translate`

Shared helpers:

- `_run_text_translate(text, ...)` — auto pass + optional hint pass + optional analyze + send + collect; used by both inline and file text handlers
- `_collect_text(result, ...)` — non-fatal `save_text_submission` + logging; called from `_run_text_translate`
- `_check_content_safety(raw, content_type, filename)` — returns error string or None; null-byte check for `text/*`, magic-byte check for `image/*` and `audio/*`
- `_fetch_header(url, n=16)` — lightweight aiohttp range request to get the first 16 bytes of an image or audio URL for magic-byte validation without a full download
- `_MAGIC_BYTES` — dict mapping MIME type → list of valid leading byte sequences; covers PNG, JPEG, GIF, WebP, BMP, OGG, MP3, WAV, WebM
- Size limits: `_MAX_IMAGE_BYTES = 8 MB`, `_MAX_AUDIO_BYTES = 8 MB`, `_MAX_TEXT_BYTES = 50 KB`, `_MAX_TRANSLATE_CHARS = 3000` (text content cap before translation)

## Architecture

### Text Translation (`Translation/1-Text/`)

- **`detect.py`** — Language detection and text segmentation:
  - `segment_text(text)` — single unified segmentation function used by both the translation pipeline and the bot's `--analyze` display. Identifies CJK spans by Unicode range, classifies Latin gaps via lingua (noise-stripped: @mentions and digits removed before detection).
  - CJK language identification: Hangul ranges → Korean unambiguously; Kana ranges → Japanese unambiguously; pure CJK Unified Ideographs → langdetect zh/ja disambiguation with Korean veto when no Hangul present.
  - `analyze_segments(text)` — thin wrapper over `segment_text` that groups spans by language name for display.
  - `detect_language_with_confidence(text)` — strips Latin script before running langdetect so foreign content is not diluted.

- **`translate_text.py`** — Translation with local-model-first, HF API fallback:
  - Default models: `Helsinki-NLP/opus-mt-mul-en` (any → English), `Helsinki-NLP/opus-mt-en-{tgt}` (English → target).
  - `_LANG_MODEL_OVERRIDES` — dict of per-language model replacements for cases where the default pattern is unavailable or produces poor output. Currently covers Korean: `ko → (opus-mt-ko-en, opus-mt-tc-big-en-ko)`. Add new entries here to extend coverage without touching call sites.
  - Model resolution order (both directions): local fine-tuned model → override HF model → default pattern model.
  - `_translate_to_english(text, client)` and `_translate_from_english(text, tgt_lang, client)` are private; `translate_text()` is the public API.
  - `_translate_via_segments(segs, client, src_lang)` — translates each non-English segment in place, leaving English spans unchanged. Single code path for all mixed-script cases.
  - Same-language passthrough: when auto-detected source base code matches the requested non-English target, returns `method="passthrough"` with original text unchanged — prevents the two-hop zh→en→zh mangling bug.
  - Local model loading: `_load_local_model(direction)` checks `~/.tl-bot/models/<direction>/` for a fine-tuned MarianMT model (installed by `deploy.py`). Cached at module level. Falls back to HF Inference API if not present.
  - Returns dict: `{translated_text, source_language, confidence, method}`. Method values: `none` | `passthrough` | `opus-mt` | `opus-mt-segmented`.

- **`utils.py`** — Language code/name mappings: `get_language_name()`, `get_mbart_code()`, `parse_language_hint()`, and alias table `_HINT_ALIASES`.

### Audio Transcription & Translation (`Translation/3-Audio/`)

- **`transcribe_audio.py`** — Transcription via HuggingFace Inference API (Whisper):
  - `transcribe(source, src_lang=None)` — public API; accepts bytes, file path, or URL string.
  - Calls `openai/whisper-large-v3` via `InferenceClient.automatic_speech_recognition()`.
  - Language detection on the transcript reuses `detect_language_with_confidence()` from `detect.py` — no separate model needed.
  - Returns dict: `{transcript, source_language, confidence, method}`. Method value: `whisper-hf`.
  - **Whisper hallucination**: Whisper always returns some text even for silence/noise; the bot's "no speech" guard is only reached if the API returns an empty string.
  - See `WHISPER_LOCAL.md` for the migration path to a locally fine-tuned model.

- **`translate_audio.py`** — Full audio pipeline (transcribe → translate → collect):
  - `translate_audio(source, from_lang, to_lang, filename, username)` — public API.
  - Result dict uses `original_text` (the transcript) keyed to match `_fmt_result(ocr=True)` so transcript + translation display reuses the existing formatter.
  - Returns `{original_text, translated_text, source_language, confidence, method, collected}`.
  - Collection is non-fatal (bare `except Exception: pass`).

- **`WHISPER_LOCAL.md`** — Step-by-step guide for migrating to a locally fine-tuned Whisper model: when to switch, dependencies, local-first inference pattern (matching `translate_text.py`), fine-tuning with `Seq2SeqTrainer`, deploy to `~/.tl-bot/whisper/`, corrections schema.

### Video Translation & Audio Extraction (`Translation/4-Video/`)

- **`extract_audio.py`** — Extracts the audio track from a video file using PyAV:
  - `extract_audio(source)` — public API; accepts bytes, local file path (str or Path), or URL string.
  - Decodes the first audio stream and resamples to 16 kHz mono WAV using `av.AudioResampler(format="s16", layout="mono", rate=16000)`.
  - `_write_wav(pcm, sample_rate)` — manually constructs a RIFF/WAVE header + int16 PCM body; no external WAV library needed.
  - PyAV bundles its own compiled FFmpeg libraries — **no system ffmpeg binary required**.
  - Raises `RuntimeError` if no audio stream is found or decoding produces no frames.

- **`translate_video.py`** — Full video pipeline (extract audio → transcribe → translate → collect):
  - `translate_video(source, from_lang, to_lang, filename, username)` — public API.
  - Delegates to `extract_audio` → `transcribe` (Whisper via 3-Audio) → `translate_text` (via 1-Text).
  - Result dict uses `original_text` (the transcript) keyed to match `_fmt_result(ocr=True)`.
  - Returns `{original_text, translated_text, source_language, confidence, method, collected}`.
  - Collection via `collect_video.save_submission()` is non-fatal.

### Image Translation (`Translation/2-Image/`)

- **`ocr.py`** — EasyOCR pipeline with OpenCV preprocessing:
  - Three lazy EasyOCR readers (zh/ja/ko, each paired with en). Best average confidence wins on auto-detect; specific reader used when hinted.
  - Post-processing: `_split_merged_words()` — splits letter↔digit boundaries (`_DIGIT_BOUNDARY_RE`) then applies wordninja to any 15+ character pure-Latin run to recover merged words.
  - Six preprocessing variants (`preprocess` is the production default):
    - `preprocess()` — baseline: 2× INTER_CUBIC + grayscale
    - `preprocess_enhanced()` — LANCZOS4 + denoise(h=5) + CLAHE(clipLimit=1.0, 8×8) + unsharp mask (1.3/−0.3)
    - `preprocess_discord()` — enhanced + dark-mode inversion (checked on raw gray before CLAHE, p75 < 128) + stripe removal (light mode only, scaled height ≥ 80px)
    - `preprocess_otsu()` — baseline + Otsu binarization; degrades on complex images, do not use as default
    - `preprocess_light_denoise()` — LANCZOS4 + lighter denoise(h=2) + CLAHE(4×4) + unsharp; most consistent challenger to baseline
    - `preprocess_bilateral()` — LANCZOS4 + bilateral filter(d=9) + CLAHE(4×4) + unsharp; strong on clean images, degrades on complex screenshots
  - `extract_text(source)` — returns sorted segment list `[{text, confidence, bbox}]`
  - `extract_text_combined(source)` — returns `(combined_text, avg_confidence)`
  - `extract_text_hinted(processed, lang_code, read_kwargs)` — uses script-appropriate reader

- **`translate_image.py`** — Full image pipeline (load → OCR → detect → translate):
  - Dual-pass when `--from` is given: auto pass (best-confidence reader) + hinted pass (script-targeted reader). Both returned as `{auto, hint}`.
  - Score = `ocr_confidence × lang_confidence` for comparison.
  - After each auto pass, calls `collect_image.save_submission()` (imported from `0-Data/Image/training/`) to store the image + labels for training (non-fatal if it fails).
  - Production default uses `preprocess()` (baseline).

### Data Collection & Training (`Translation/0-Data/`)

All collected data, training pipelines, and dev/testing tools live here, separated from the active translation packages.

```
Translation/0-Data/
  Image/
    data/                       ← flat image store (no subdirectory)
      *.png                     ← raw Discord-submitted images
      labels.jsonl              ← OCR labels + correction fields
      preprocess_comparison/    ← variant PNGs from compare_preprocess.py
      demo_output/              ← synthetic PNGs from demo.py
    training/
      collect_image.py          ← save_submission(), dataset_stats()
      dataset.py                ← build LMDB with optional augmentation
      train.py                  ← additive fine-tuning (None-VGG-BiLSTM-CTC)
      deploy.py                 ← installs weights into ~/.EasyOCR/
      tl_bot_ocr.py             ← model definition
      tl_bot_ocr.yaml           ← architecture config (character set, imgH/W)
      review.py                 ← interactive correction CLI
    testing/
      compare_preprocess.py     ← side-by-side variant comparison across all 6 variants
      demo.py                   ← full pipeline demo (PIL synthetic images → OCR → translate → collect)
  Text/
    data/
      text_submissions.jsonl    ← translated text submissions + correction fields
    training/
      collect_text.py           ← save_submission(), dataset_stats()
      dataset.py                ← export (source, target) pairs to JSONL for training
      train.py                  ← seq2seq fine-tuning (MarianMT via Seq2SeqTrainer)
      deploy.py                 ← installs fine-tuned model to ~/.tl-bot/models/
    testing/
      demo.py                   ← full text pipeline demo (sample texts → translate → collect → report)
  Audio/
    data/
      audio_submissions.jsonl   ← transcription + translation submissions + correction fields
    training/
      collect_audio.py          ← save_submission(), dataset_stats()
      (dataset.py / train.py / deploy.py — not yet created; see Translation/3-Audio/WHISPER_LOCAL.md)
  Video/
    data/
      video_submissions.jsonl   ← transcript + translation submissions + correction fields
    training/
      collect_video.py          ← save_submission(), dataset_stats(), load_submissions()
    testing/
      demo.py                   ← full video pipeline demo (gTTS → MKV → extract → transcribe → translate → collect)
```

#### `labels.jsonl` schema

```jsonc
{
  "filename": "20260614_username_stem.png",
  "image_hash": "<sha1[:16]>",
  "ocr_text": "<bot OCR output>",
  "correct_text": null,         // manual override used by dataset.py over ocr_text
  "correct_translation": null,  // manually verified translation (reserved for future training)
  "source_language": "zh-cn",
  "confidence": 1.0,
  "ocr_confidence": 0.87,
  "username": "discorduser",
  "timestamp": "2026-..."
}
```

Whether a correction exists is inferred from data — `correct_text != null` means a human override is in effect. No separate `reviewed` flag.

#### `text_submissions.jsonl` schema

```jsonc
{
  "text_hash": "<sha1[:16]>",
  "original_text": "<source text>",
  "translated_text": "<bot translation>",
  "correct_translation": null,  // manual override
  "source_language": "ko",
  "target_language": "en",
  "confidence": 0.99,
  "method": "opus-mt-segmented",
  "username": "discorduser",
  "timestamp": "2026-..."
}
```

#### `audio_submissions.jsonl` schema

```jsonc
{
  "text_hash": "<sha1[:16] of transcript>",
  "filename": "voice-message.ogg",
  "transcript": "<Whisper output>",
  "correct_transcript": null,     // manual ASR correction; primary training signal for Whisper fine-tune
  "translated_text": "<translation>",
  "correct_translation": null,    // manual translation correction
  "source_language": "ko",
  "target_language": "en",
  "confidence": 0.95,
  "method": "whisper-hf",         // 'whisper-hf' | 'whisper-local' (after local deploy)
  "username": "discorduser",
  "timestamp": "2026-..."
}
```

Deduplication is by SHA-1 of the transcript text. `correct_transcript != null` signals an ASR correction; `correct_translation != null` signals a translation correction.

#### `video_submissions.jsonl` schema

```jsonc
{
  "text_hash": "<sha1[:16] of transcript>",
  "filename": "clip.mp4",
  "transcript": "<Whisper output>",
  "correct_transcript": null,     // manual ASR correction
  "translated_text": "<translation>",
  "correct_translation": null,    // manual translation correction
  "source_language": "ko",
  "target_language": "en",
  "confidence": 0.95,
  "method": "whisper-hf",         // 'whisper-hf' | 'whisper-local' (after local deploy)
  "username": "discorduser",
  "timestamp": "2026-..."
}
```

Deduplication is by SHA-1 of the transcript text, same as audio. Schema is identical to `audio_submissions.jsonl` except `filename` references the original video file rather than an audio file.

#### Text training workflow

```bash
cd Translation/0-Data/Text/training/

# 1. Review and correct translation labels (review.py lives in Image/training/ — covers both modalities)
.venv\Scripts\python.exe Translation\0-Data\Image\training\review.py --mode text
.venv\Scripts\python.exe Translation\0-Data\Image\training\review.py --stats

# 2. Build dataset (source/target JSONL pairs)
.venv\Scripts\python.exe dataset.py --split 0.9              # all language pairs
.venv\Scripts\python.exe dataset.py --split 0.9 --src ko     # Korean-only pairs

# 3. Fine-tune
.venv\Scripts\python.exe train.py --direction mul-en --epochs 5
.venv\Scripts\python.exe train.py --direction en-ko --epochs 5

# 4. Deploy to ~/.tl-bot/models/
.venv\Scripts\python.exe deploy.py --direction mul-en
.venv\Scripts\python.exe deploy.py --direction en-ko
.venv\Scripts\python.exe deploy.py --list    # verify installed
# translate_text.py automatically prefers local model over HF API once deployed
```

Pair directions:
- `mul-en`: `(original_text → translated_text/correct_translation)` from submissions where `target_language == "en"`. Fine-tunes `opus-mt-mul-en`.
- `en-{tgt}`: synthesized by reversing mul-en pairs — `(english_translation → original_text)` grouped by source language. Fine-tunes `opus-mt-en-{tgt}` per language.

Local inference: fine-tuned models load from `~/.tl-bot/models/<direction>/` via `MarianMTModel.from_pretrained`. Models are cached at module level. HF Inference API is used as fallback when a direction has no local model deployed.

#### Text testing tools

```bash
cd Translation/0-Data/Text/testing/

# Full pipeline demo (runs sample Korean/Chinese/Japanese/English texts)
.venv\Scripts\python.exe demo.py
.venv\Scripts\python.exe demo.py --no-collect   # skip saving to data/
.venv\Scripts\python.exe demo.py --tgt fr        # translate to French
```

#### Image training workflow

```bash
cd Translation/0-Data/Image/training/

# 1. Review and correct OCR labels (optional but improves quality)
.venv\Scripts\python.exe review.py              # image OCR corrections
.venv\Scripts\python.exe review.py --mode text  # text translation corrections
.venv\Scripts\python.exe review.py --stats      # coverage report

# 2. Build LMDB dataset
.venv\Scripts\python.exe dataset.py --split 0.9                        # no augmentation
.venv\Scripts\python.exe dataset.py --split 0.9 --augment              # 3× augmented copies (default)
.venv\Scripts\python.exe dataset.py --split 0.9 --augment --aug-factor 4 --aug-ops jpeg,noise,blur

# 3. Fine-tune
.venv\Scripts\python.exe train.py --epochs 10 --batch 32

# 4. Deploy to EasyOCR
.venv\Scripts\python.exe deploy.py
# After deploy, update ocr.py readers to pass recog_network="tl_bot_ocr"
```

Augmentation ops: `brightness`, `contrast`, `noise`, `blur`, `jpeg`, `rotate`. Augmentation is only applied to the train split, never val. `dataset.py` prints how many entries used `correct_text` vs raw `ocr_text`.

The fine-tuned model targets **Latin character recognition within CJK images** (usernames, UI text, mixed content). Pure CJK recognition uses EasyOCR's built-in CJK models and is unaffected. To activate the fine-tuned model after `deploy.py`, pass `recog_network="tl_bot_ocr"` to the `easyocr.Reader(...)` calls in `ocr.py`.

#### Dev/testing tools

```bash
cd Translation/0-Data/Image/testing/

# Compare all 6 preprocessing variants on collected images
.venv\Scripts\python.exe compare_preprocess.py
.venv\Scripts\python.exe compare_preprocess.py --images path/to/dir

# Full pipeline demo (generates synthetic CJK/mixed/English images)
.venv\Scripts\python.exe demo.py
.venv\Scripts\python.exe demo.py --save-images   # write PNGs to data/demo_output/
.venv\Scripts\python.exe demo.py --no-collect    # skip saving to data/
```

#### Video testing tools

```bash
cd Translation/0-Data/Video/testing/

# Full pipeline demo (synthesizes speech via gTTS, wraps in MKV, runs full video pipeline)
.venv\Scripts\python.exe demo.py
.venv\Scripts\python.exe demo.py --no-collect     # skip saving to data/
.venv\Scripts\python.exe demo.py --tgt fr         # translate to French
.venv\Scripts\python.exe demo.py --save-videos    # write generated MKVs to data/demo_output/
```

Requires internet access (gTTS for speech synthesis + HF Whisper API for transcription). `wrap_audio_in_mkv()` uses PyAV to encode gTTS MP3 into a Matroska container using the libopus codec at 48 kHz — no system ffmpeg needed.

### Research Notebooks

Exploratory work; production pipelines are in the `.py` modules above:

- `Translation/2-Image/backup/OCR_Models.ipynb` — EasyOCR, pytesseract, TrOCR approaches
- `Translation/1-Text/Text_Translation.ipynb` — mBART-50, Qwen2.5 experiments (HF token must come from environment, not hardcoded)
- `Translation/3-Audio/Audio_Translation.ipynb` — original speech-to-text research; production pipeline is now in `transcribe_audio.py` / `translate_audio.py`

### Typography Module (`Typography/`)

`get_fonts.py` scans Windows system fonts, extracts metadata (family, weight, italic, serif, Unicode CMAP coverage), renders glyph previews with PIL, and organises output into a font dataset grouped by Unicode block. **Windows-specific** (hardcoded `C:/Windows/Fonts`). CLI: `--update`, `--style-folders`. Output dirs (`font_data/`, `font-dataset/`, `windows-fonts/`) are gitignored.

`Typography/Typography_Model.ipynb` — PyTorch font classification model.

## Dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages: `discord.py`, `python-dotenv`, `easyocr`, `opencv-python`, `wordninja`, `langdetect`, `lingua-language-detector`, `huggingface_hub`, `transformers`, `torch`, `pillow`, `fonttools`, `pytest`, `av` (PyAV — bundles its own FFmpeg libs, no system ffmpeg needed).

`lmdb` is required only for the training pipeline (`pip install lmdb`); it is not in `requirements.txt`.

Always run scripts with `.venv\Scripts\python.exe` — the system Python lacks `cv2`, `easyocr`, and other packages.

## Key Conventions

- **Intents**: `message_content` intent is enabled — must also be enabled in the Discord Developer Portal.
- **Secrets**: `DISCORD_BOT_TOKEN` and `HF_TOKEN` must never be hardcoded; load from `.env` (gitignored).
- **No git co-author tags**: Do not add `Co-Authored-By: Claude` lines to commits in this repo.
- **Gitignored data dirs**: `Translation/0-Data/Image/data/`, `Translation/0-Data/Text/data/`, `Translation/0-Data/Audio/data/`, `Translation/0-Data/Image/training/checkpoints/`, `font_data/`, `font-dataset/`, `windows-fonts/` — don't commit collected images, audio files, JSONL datasets, LMDB files, or model checkpoints.
- **Test suite**: `pytest` tests exist under `Translation/1-Text/tests/`, `Translation/2-Image/tests/`, `Translation/3-Audio/tests/`, and `Translation/4-Video/tests/`. Run with `pytest` from the repo root. All four suites use mocks — no network calls required. (84 tests pass, 1 skips if `test_image.png` is absent.)
- **Preprocessing variants**: All six variants are available in `ocr.py`; `preprocess()` (baseline) is the production default. Use `compare_preprocess.py` to evaluate before switching. `light_denoise` is the most consistent alternative.
- **Public API surface**: `translate_text()` in `translate_text.py` is the public entry point; `_translate_to_english()` and `_translate_from_english()` are private implementation details.
- **Collection is non-fatal**: `collect_image.save_submission()`, `collect_text.save_submission()`, and `collect_audio.save_submission()` are all wrapped in try/except — a collection failure must never prevent the translation response from being sent.
