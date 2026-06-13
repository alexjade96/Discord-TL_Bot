# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discord bot (`TL-Bot.py`) that translates text and images for users. The bot is triggered by @mention and routes `/translate` commands based on attachment type (image, audio, video, text) or inline text. Image and text translation pipelines are fully wired into the bot; audio/video stubs remain.

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
- `/translate` + audio/video/text attachment → stub (not yet implemented)
- Unrecognized command → fallback message

Flags parsed by `_parse_translate_flags()` in `TL-Bot.py`:
- `--from <language>` — hint the source language; triggers a dual-pass comparison for images
- `--to <language>` — set output language (default: English)
- `--analyze` — show segment analysis alongside the translation

Language names/codes are resolved via `parse_language_hint()` in `Translation/2-Text/utils.py` (supports common names, aliases, ISO 639-1 codes).

## Architecture

### Text Translation (`Translation/2-Text/`)

- **`detect.py`** — Language detection and text segmentation:
  - `segment_text(text)` — single unified segmentation function used by both the translation pipeline and the bot's `--analyze` display. Identifies CJK spans by Unicode range, classifies Latin gaps via lingua (noise-stripped: @mentions and digits removed before detection).
  - CJK language identification: Hangul ranges → Korean unambiguously; Kana ranges → Japanese unambiguously; pure CJK Unified Ideographs → langdetect zh/ja disambiguation with Korean veto when no Hangul present.
  - `analyze_segments(text)` — thin wrapper over `segment_text` that groups spans by language name for display.
  - `detect_language_with_confidence(text)` — strips Latin script before running langdetect so foreign content is not diluted.

- **`translate_text.py`** — Translation via HuggingFace Inference API:
  - Model: `Helsinki-NLP/opus-mt-mul-en` (multilingual → English)
  - Reverse model: `Helsinki-NLP/opus-mt-en-{tgt}` (English → target)
  - `_translate_via_segments(segs, client, src_lang)` — translates each non-English segment in place, leaving English spans unchanged. Single code path for all mixed-script cases.
  - Returns dict: `{translated_text, source_language, confidence, method}`. Method values: `none` | `passthrough` | `opus-mt` | `opus-mt-segmented`.

- **`utils.py`** — Language code/name mappings: `get_language_name()`, `get_mbart_code()`, `parse_language_hint()`, and alias table `_HINT_ALIASES`.

### Image Translation (`Translation/1-Image/`)

- **`ocr.py`** — EasyOCR pipeline with OpenCV preprocessing:
  - Three lazy EasyOCR readers (zh/ja/ko, each paired with en). Best average confidence wins on auto-detect; specific reader used when hinted.
  - Post-processing: `_split_merged_words()` — splits letter↔digit boundaries (`_DIGIT_BOUNDARY_RE`) then applies wordninja to any 15+ character pure-Latin run to recover merged words.
  - Three preprocessing variants (none wired as production default):
    - `preprocess()` — baseline: 2× INTER_CUBIC + grayscale
    - `preprocess_enhanced()` — LANCZOS4 + denoise(h=5) + CLAHE(clipLimit=1.0) + unsharp mask (1.3/−0.3)
    - `preprocess_discord()` — enhanced + dark-mode inversion (checked on raw gray **before** CLAHE, p75 < 128) + stripe removal (light mode only, scaled height ≥ 80px)
  - `extract_text(source)` — returns sorted segment list `[{text, confidence, bbox}]`
  - `extract_text_combined(source)` — returns `(combined_text, avg_confidence)`
  - `extract_text_hinted(processed, lang_code, read_kwargs)` — uses script-appropriate reader

- **`translate_image.py`** — Full image pipeline (load → OCR → detect → translate):
  - Dual-pass when `--from` is given: auto pass (best-confidence reader) + hinted pass (script-targeted reader). Both returned as `{auto, hint}`.
  - Score = `ocr_confidence × lang_confidence` for comparison.
  - After each auto pass, calls `collect.save_submission()` to store the image + labels for training (non-fatal if it fails).
  - Production default uses `preprocess()` (baseline).

- **`collect.py`** — Training data collection:
  - Saves raw Discord-submitted images to `data/images/` and appends metadata to `data/labels.jsonl`.
  - Deduplicates by SHA-1 of raw image bytes (first 16 hex chars).
  - `dataset_stats()` — prints language breakdown of collected data.

- **`compare_preprocess.py`** — Dev tool for comparing preprocessing variants against test images in `data/images/`. Saves processed PNGs to `data/preprocess_comparison/`. Not wired into the bot.

- **`demo.py`** — Full pipeline demonstration using PIL-generated synthetic text images (CJK, mixed, English). Runs OCR → translation → collection and prints a pass/fail summary. Windows-specific font paths (`C:/Windows/Fonts`).

- **`training/`** — Fine-tuning pipeline for the EasyOCR recognition model:
  - `dataset.py` — builds LMDB datasets from `data/labels.jsonl`
  - `tl_bot_ocr.py` — None-VGG-BiLSTM-CTC model definition
  - `train.py` — additive fine-tuning (continues from `checkpoints/tl_bot_ocr_latest.pth` if present)
  - `deploy.py` — installs trained weights into `~/.EasyOCR/` for inference
  - Workflow: `python dataset.py --split 0.9` → `python train.py` → `python deploy.py`

### Research Notebooks

These document exploratory work; the production pipelines are in the `.py` modules above:

- `Translation/1-Image/OCR_Models.ipynb` — EasyOCR, pytesseract, TrOCR approaches
- `Translation/2-Text/Text_Translation.ipynb` — mBART-50, Qwen2.5 experiments (HF token must come from environment, not hardcoded)
- `Translation/3-Audio/Audio_Translation.ipynb` — speech-to-text + translation (not yet integrated)

### Typography Module (`Typography/`)

`get_fonts.py` scans Windows system fonts, extracts metadata (family, weight, italic, serif, Unicode CMAP coverage), renders glyph previews with PIL, and organises output into a font dataset grouped by Unicode block. **Windows-specific** (hardcoded `C:/Windows/Fonts`). CLI: `--update`, `--style-folders`. Output dirs (`font_data/`, `font-dataset/`, `windows-fonts/`) are gitignored.

`Typography/Typography_Model.ipynb` — PyTorch font classification model.

## Dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages: `discord.py`, `python-dotenv`, `easyocr`, `opencv-python`, `wordninja`, `langdetect`, `lingua-language-detector`, `huggingface_hub`, `transformers`, `torch`, `pillow`, `fonttools`, `pytest`.

`lmdb` is required only for the training pipeline (`pip install lmdb`); it is not in `requirements.txt`.

## Key Conventions

- **Intents**: `message_content` intent is enabled — must also be enabled in the Discord Developer Portal.
- **Secrets**: `DISCORD_BOT_TOKEN` and `HF_TOKEN` must never be hardcoded; load from `.env` (gitignored).
- **No git co-author tags**: Do not add `Co-Authored-By: Claude` lines to commits in this repo.
- **Gitignored data dirs**: `data/`, `font_data/`, `font-dataset/`, `windows-fonts/`, `checkpoints/` — don't commit model outputs, collected images, or font datasets.
- **Test suite**: `pytest` tests exist under `Translation/1-Image/tests/` and `Translation/2-Text/tests/`. Run with `pytest` from the repo root.
- **Preprocessing variants**: All three (`preprocess`, `preprocess_enhanced`, `preprocess_discord`) are available in `ocr.py` but none replaces the production default (`preprocess`). Compare with `compare_preprocess.py` before switching.
