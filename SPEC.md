# Discord TL-Bot — Project Specification

## Goal

Build a Discord bot that accepts text, images, and audio from users and returns translated text in English. Translation pipelines are powered by HuggingFace models, either via the Inference API (remote) or loaded locally with `transformers`.

---

## Bot Interaction Model

The bot is invoked by @mention. Everything after the mention is the command string.

| User input | Bot behavior |
|---|---|
| `@TL-Bot` (no args) | Greeting with usage instructions |
| `@TL-Bot /help` | List available commands |
| `@TL-Bot /translate <text>` | Translate inline text |
| `@TL-Bot /translate` + image attachment | Run OCR pipeline then translate |
| `@TL-Bot /translate` + audio attachment | Run audio transcription pipeline then translate |
| `@TL-Bot /translate` + text file attachment | Read file contents and translate |
| Unrecognized command | Fallback message |

Responses are sent to the same channel as the original message. If a pipeline takes more than a few seconds, the bot should send an interim "Processing…" reply.

---

## Translation Pipelines

### 1. Text Translation

**Input:** A plain-text string (inline from the command, or extracted from a prior pipeline step).  
**Output:** The translated English string, plus the detected source language.

Steps:
1. **Language detection** — run `langdetect` first for a quick check; fall back to `lingua` (per-segment confidence) if the input is short or ambiguous.
2. **Skip translation** if detected language is already English.
3. **Translate** using `facebook/mbart-large-50-many-to-many-mmt` via the HuggingFace Inference API (`huggingface_hub.InferenceClient`).  
   - Alternative/fallback: prompt `Qwen/Qwen2.5-1.5B-Instruct` with an instruction template when mBART does not support the source language.
4. Return the translated text and source language label to Discord.

Language codes for mBART follow the `xx_XX` format (e.g., `zh_CN`, `ja_XX`, `ko_KR`, `ar_AR`). Map `langdetect` codes to mBART codes before calling the API.

---

### 2. Image Translation (OCR → Text → Translate)

**Input:** An image attachment (PNG, JPG, WEBP).  
**Output:** Translated English text extracted from the image.

Steps:
1. **Download** the attachment using its Discord URL.
2. **Preprocess** with OpenCV: scale 2×, convert to grayscale.
3. **OCR** with EasyOCR (`easyocr.Reader(['en', 'ch_sim', 'ja', 'ko'])`) — preferred because it handles CJK scripts without OS-level font dependencies.  
   - Fallback: `pytesseract` with config `-l eng+jpn+chi_sim+chi_tra+kor --oem 3 --psm 3` (requires the Tesseract binary).
4. **Aggregate** text segments from OCR output, preserving reading order (top-to-bottom, left-to-right based on bounding box coordinates).
5. **Pass the aggregated text** through the Text Translation pipeline (step 1 above).
6. Return translated text to Discord. If no text was found, reply with a message indicating the image contained no readable text.

---

### 3. Audio Translation (ASR → Text → Translate)

**Input:** An audio attachment (MP3, WAV, OGG, M4A) or video attachment (MP4, WEBM — extract audio track).  
**Output:** Translated English transcript.

Steps:
1. **Download** the attachment.
2. **Transcribe** using the HuggingFace Inference API with an automatic speech recognition (ASR) model. Preferred model: `openai/whisper-large-v3` (multilingual, handles CJK and European languages). Fallback: `facebook/wav2vec2-base-960h` (English only).
3. **Pass the transcript** through the Text Translation pipeline.
4. Return the translated transcript to Discord, prefixed with the detected source language.

For video attachments, extract the audio stream using `ffmpeg` before passing to ASR.

---

## HuggingFace Integration

All model calls go through `huggingface_hub.InferenceClient`. The HuggingFace API token must be loaded from the environment variable `HF_TOKEN` — never hardcoded.

```python
from huggingface_hub import InferenceClient
client = InferenceClient(token=os.getenv("HF_TOKEN"))
```

| Task | Model ID | API method |
|---|---|---|
| Text translation | `facebook/mbart-large-50-many-to-many-mmt` | `client.translation()` |
| Text generation (fallback) | `Qwen/Qwen2.5-1.5B-Instruct` | `client.text_generation()` |
| Speech recognition | `openai/whisper-large-v3` | `client.automatic_speech_recognition()` |
| Language detection | `langdetect` + `lingua` (local libraries, no API call) | — |

---

## Configuration

All secrets and configurable values must come from environment variables (loaded via a `.env` file, which is gitignored).

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | Discord bot token |
| `HF_TOKEN` | Yes | HuggingFace Inference API token |

Required Discord Gateway Intents (must also be enabled in the Discord Developer Portal):
- `message_content` — to read message text
- Default intents (guilds, messages)

---

## Error Handling

- If an attachment download fails, reply with an error message and log the HTTP status.
- If OCR returns no text, reply with "No readable text found in the image."
- If language detection is inconclusive, attempt translation anyway and note the uncertainty in the reply.
- If the HuggingFace API returns an error, reply with a user-friendly message and log the full error.
- All errors should be logged to the rotating log file at `./logs/TL_Bot_<date>.log`.

---

## Module Structure (target)

Integrate the notebook prototypes into importable Python modules:

```
TL-Bot.py                  # Discord client, event routing
translation/
    text.py                # detect_language(), translate_text()
    image.py               # ocr_image(), extract_text_from_image()
    audio.py               # transcribe_audio()
    utils.py               # download_attachment(), langdetect_to_mbart_code()
```

Each module should be independently runnable for testing without starting the Discord bot.

---

## Out of Scope

- Translation into languages other than English (target is always English for now).
- Real-time voice channel transcription.
- Storing or logging user message content beyond what `discord.py` logs by default.
- The Typography / font-rendering module (`Typography/`) — independent utility, not part of the bot.
