# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discord bot (`TL-Bot.py`) that translates text, images, and audio for users. The bot is triggered by @mention in Discord and routes `/translate` commands to the appropriate pipeline based on attachment type (image, audio, video, text) or inline text.

## Running the Bot

```bash
python TL-Bot.py
```

Logs are written to `./logs/TL_Bot_<date>.log` (rotating, 32 MiB max, 5 backups). The `logs/` directory is auto-created at startup.

## Bot Token

The bot token **must not be hardcoded** in `TL-Bot.py`. It should be loaded from an environment variable or `.env` file (`.env` is gitignored). Replace line 99 with:

```python
bot_token = os.getenv("DISCORD_BOT_TOKEN")
```

## Architecture

### Bot Interaction Model (`TL-Bot.py`)

The bot listens for all messages via `on_message` but only acts when `@TL-Bot` is the first token. The remaining message content (everything after the mention) is parsed as a command string. Current command routing:

- No args → greeting
- `/help` → command list
- `/translate` + attachment → routes by `attachment.content_type` (`image/*`, `audio/*`, `video/*`, `text/*`)
- `/translate` + inline text → passes text for translation
- Unrecognized → fallback message

All translation logic stubs are currently `# Here you would add...` comments — the pipelines are not yet wired into the bot.

### Translation Pipelines (Jupyter Notebooks)

The three translation modules exist as research notebooks, not yet integrated into the bot:

- **`Translation/1-Image/OCR_Models.ipynb`** — Three OCR approaches explored: EasyOCR (multi-language, preferred for CJK), pytesseract (Tesseract wrapper, Windows path assumed), and TrOCR (Microsoft vision encoder-decoder, commented out). Language detection uses `langdetect` post-OCR.

- **`Translation/2-Text/Text_Translation.ipynb`** — Language detection via `langdetect` and `lingua` (per-segment confidence). Translation via `facebook/mbart-large-50-many-to-many-mmt` (HuggingFace Inference API) and `Qwen/Qwen2.5-1.5B-Instruct`. HuggingFace API token must come from environment (not hardcoded as in notebooks).

- **`Translation/3-Audio/Audio_Translation.ipynb`** — Audio transcription and translation pipeline (speech-to-text + translation).

### Typography Module (`Typography/`)

`get_fonts.py` is a standalone utility that scans Windows system fonts, extracts metadata (family, weight, italic, serif classification, Unicode CMAP coverage), renders glyph previews with PIL, and organizes output into a font dataset grouped by Unicode block (Latin, CJK, Arabic, Devanagari, etc.). It is **Windows-specific** (hardcoded path to `C:/Windows/Fonts`). CLI flags: `--update`, `--style-folders`. Output directories (`font_data/`, `font-dataset/`, `windows-fonts/`) are gitignored.

`Typography/Typography_Model.ipynb` contains a PyTorch model for font classification/processing.

## Dependencies

There is no `requirements.txt`. Install dependencies manually based on what the code uses:

```bash
pip install discord.py easyocr pytesseract opencv-python pillow transformers torch langdetect lingua-language-detector fonttools tqdm
```

Tesseract OCR binary must also be installed separately (e.g., `apt install tesseract-ocr` on Linux).

## Key Conventions

- **Intents**: `message_content` intent is enabled — required for reading message text. This must be enabled in the Discord Developer Portal for the bot application.
- **Gitignored data dirs**: `data/`, `font_data/`, `font-dataset/`, `windows-fonts/`, `checkpoints/` — don't commit model outputs or font datasets.
- **No test suite**: Validation is done manually using the test images in `Translation/1-Image/`.
