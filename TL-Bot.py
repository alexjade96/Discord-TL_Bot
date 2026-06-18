# TL-Bot.py
# Discord bot for handling image/text translations

import asyncio
import os
import sys
import datetime
from pathlib import Path

import aiohttp
import discord
import logging
import logging.handlers

# Load .env file if python-dotenv is installed
from dotenv import load_dotenv
load_dotenv()

# Add text translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "1-Text"))
from translate_text import translate_text
from utils import get_language_name, parse_language_hint  # noqa: E402
from detect import analyze_segments  # noqa: E402

# Add image translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "2-Image"))
from translate_image import translate_image  # noqa: E402
from ocr import extract_text_combined  # noqa: E402

# Add audio translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "3-Audio"))
from translate_audio import translate_audio  # noqa: E402

# Add video translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "4-Video"))
from translate_video import translate_video  # noqa: E402

# Text collection
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "0-Data" / "Text" / "training"))
from collect_text import save_submission as save_text_submission  # noqa: E402

# Logging handler setup
logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
logging.getLogger("discord.http").setLevel(logging.INFO)

today = datetime.datetime.now().strftime("%Y-%m-%d")
logdir = Path("logs")
os.makedirs(logdir, exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    filename=f"{logdir}/TL_Bot_{today}.log",
    encoding="utf-8",
    maxBytes=32 * 1024 * 1024,  # 32 MiB
    backupCount=5,
)
dt_fmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(
    "[{asctime}] [{levelname:<8}] {name}: {message}", dt_fmt, style="{"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Intents
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# Magic byte signatures for common image formats.
# Used by _check_content_safety to detect files whose bytes don't match their
# declared content type (e.g. an executable uploaded as image/png).
_MAGIC_BYTES: dict[str, list[bytes]] = {
    # Images
    "image/png":  [b"\x89PNG\r\n\x1a\n"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/gif":  [b"GIF87a", b"GIF89a"],
    "image/webp": [b"RIFF"],
    "image/bmp":  [b"BM"],
    # Audio — Discord voice messages are ogg/opus; file uploads may be mp3/wav/webm
    "audio/ogg":  [b"OggS"],
    "audio/mpeg": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    "audio/wav":  [b"RIFF"],
    "audio/webm": [b"\x1a\x45\xdf\xa3"],
    # Video — MP4 ftyp box sizes vary; WebM/MKV share the EBML magic
    "video/mp4":        [b"\x00\x00\x00\x14ftyp", b"\x00\x00\x00\x18ftyp",
                         b"\x00\x00\x00\x1cftyp", b"\x00\x00\x00\x20ftyp",
                         b"\x00\x00\x00\x24ftyp", b"\x00\x00\x00\x28ftyp"],
    "video/quicktime":  [b"\x00\x00\x00\x14ftyp", b"\x00\x00\x00\x18ftyp",
                         b"\x00\x00\x00\x1cftyp", b"\x00\x00\x00\x20ftyp"],
    "video/webm":       [b"\x1a\x45\xdf\xa3"],
    "video/x-matroska": [b"\x1a\x45\xdf\xa3"],
}


def _check_content_safety(raw: bytes, content_type: str, filename: str) -> str | None:
    """Return an error message if content fails safety validation, else None.

    text/*  — rejects null bytes; plain text never contains them, so their
              presence means the file is binary data (executable, archive, etc.)
              regardless of the declared type or file extension.

    image/* — validates the first bytes against the format's magic signature;
              catches files that claim to be PNG/JPEG/etc. but aren't.

    Returns None when the content looks valid for its declared type.
    """
    ct = (content_type or "").split(";")[0].strip()
    header = raw[:16]

    if "text" in ct:
        # Scan the full content so a null byte buried past the first line is caught.
        if b"\x00" in raw:
            return f"`{filename}` appears to be a binary file, not plain text."

    if ct in _MAGIC_BYTES:
        sigs = _MAGIC_BYTES[ct]
        if not any(header.startswith(sig) for sig in sigs):
            return f"`{filename}` content does not match its declared type (`{ct}`)."

    return None


# Per-type size limits applied before downloading attachment content
_MAX_IMAGE_BYTES     = 8 * 1024 * 1024   # 8 MB
_MAX_AUDIO_BYTES     = 8 * 1024 * 1024   # 8 MB  (Discord default upload cap)
_MAX_VIDEO_BYTES     = 50 * 1024 * 1024  # 50 MB (covers Nitro Basic uploads)
_MAX_TEXT_BYTES      = 50 * 1024          # 50 KB
_MAX_TRANSLATE_CHARS = 3000               # character cap passed to translate_text


async def _fetch_header(url: str, n: int = 16) -> bytes:
    """Return the first n bytes of a URL via an HTTP Range request.

    Used to validate image magic bytes without downloading the full file.
    Returns an empty bytes object if the request fails or the server does not
    support Range requests — in that case the safety check is skipped rather
    than blocking a legitimate file.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Range": f"bytes=0-{n - 1}"}) as resp:
                if resp.status in (200, 206):
                    return await resp.read()
    except Exception:
        pass
    return b""


def _parse_translate_flags(cmd: str) -> tuple[str, str | None, str | None, bool, list[str]]:
    """Parse --from, --to, and --analyze flags out of a /translate command string.

    Returns (remaining_text, from_lang, to_lang, analyze, error_messages).
    remaining_text is the command string with all recognized flags removed.
    """
    tokens = cmd.split()
    from_lang = to_lang = None
    analyze = False
    errors: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ("--from", "--to"):
            flag = tokens[i]
            if i + 1 < len(tokens):
                parsed = parse_language_hint(tokens[i + 1])
                if parsed is None:
                    errors.append(f"Unknown language `{tokens[i + 1]}` for `{flag}`.")
                elif flag == "--from":
                    from_lang = parsed
                else:
                    to_lang = parsed
                i += 2
            else:
                errors.append(f"`{flag}` requires a language argument.")
                i += 1
        elif tokens[i] == "--analyze":
            analyze = True
            i += 1
        else:
            remaining.append(tokens[i])
            i += 1
    return " ".join(remaining), from_lang, to_lang, analyze, errors


def _is_same_language(from_lang: str | None, to_lang: str | None) -> bool:
    """Return True when source and target resolve to the same language.

    English is the implicit default target, so --from english with no --to flag
    counts as same-language. Returns False when no --from is given (nothing to check).
    """
    if from_lang is None:
        return False
    return from_lang == (to_lang or "en")


def _same_lang_msg(from_lang: str, extra: str = "") -> str:
    """Format the 'nothing to translate' message for same-language requests."""
    lang_name = get_language_name(from_lang)
    hint_tip = "To translate, add `--from <language>` and `--to <language>` (e.g. `--from chinese --to english`). Defaults to `--to English`."
    parts = [f"Source and target are both [{lang_name}] — nothing to translate.", hint_tip]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _fmt_analyze(text: str) -> str:
    """Run segment analysis on text and return a formatted Discord block."""
    segments = analyze_segments(text)
    if not segments:
        return ""
    lang_chars = {lang: sum(len(s) for s in spans) for lang, spans in segments.items()}
    total_chars = sum(lang_chars.values()) or 1
    lines = ["**Segment Analysis**"]
    for lang, spans in segments.items():
        pct = round(lang_chars[lang] / total_chars * 100)
        quoted = ", ".join(f'"{s}"' for s in spans)
        lines.append(f"{lang} ({pct}%): {quoted}")
    return "\n".join(lines)


def _fmt_result(r: dict, label: str, tgt_lang: str | None, ocr: bool = False) -> str:
    """Format a single translation result dict into a Discord message block."""
    src_name = get_language_name(r["source_language"])
    tgt_name = get_language_name(tgt_lang) if tgt_lang and tgt_lang != "en" else "English"
    conf = r.get("confidence")
    conf_str = f" ({conf * 100:.0f}%)" if conf is not None else ""
    header = f"**{label} [{src_name} → {tgt_name}]{conf_str}"
    if ocr:
        ocr_conf = r.get("ocr_confidence")
        score = r.get("score")
        ocr_str = f" | OCR {ocr_conf * 100:.0f}%" if ocr_conf is not None else ""
        score_str = f" | score {score:.2f}" if score is not None else ""
        header += f"{ocr_str}{score_str}"
    header += "**"
    body = f"> {r['original_text']}\n{r['translated_text']}" if ocr else r["translated_text"]
    return f"{header}\n{body}"


def _collect_text(
    result: dict,
    text: str,
    to_lang: str | None,
    username: str,
    filename: str = "",
) -> None:
    """Non-fatally save a text translation submission and log the outcome."""
    if result["method"] in ("none", "passthrough"):
        return
    try:
        saved = save_text_submission(
            text,
            result["translated_text"],
            source_language=result["source_language"],
            target_language=to_lang or "en",
            confidence=result.get("confidence"),
            method=result.get("method"),
            username=username,
        )
        if saved:
            if filename:
                logger.info("Collected text file: user=%s | file=%s | lang=%s | method=%s",
                            username, filename, result["source_language"], result.get("method"))
            else:
                logger.info("Collected text: user=%s | lang=%s | method=%s",
                            username, result["source_language"], result.get("method"))
        else:
            if filename:
                logger.debug("Skipped text file collection (duplicate): user=%s file=%s", username, filename)
            else:
                logger.debug("Skipped text collection (duplicate): user=%s", username)
    except Exception:
        logger.warning("Failed to save text to training dataset", exc_info=True)


async def _run_text_translate(
    text: str,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    status: discord.Message,
    username: str,
    auto_label: str = "Auto",
    hint_label: str | None = None,
    passthrough_extra: str = "",
    truncation_note: str = "",
    filename: str = "",
) -> None:
    """Run the auto+hint text translation flow and edit status with the result.

    Covers: auto pass, optional hint pass, optional segment analysis,
    passthrough detection, Discord message length capping, and collection.
    hint_label defaults to "Hint (<from_lang_name>)" when None.
    """
    auto_result = await asyncio.to_thread(translate_text, text, None, to_lang)

    if auto_result["method"] == "passthrough" and auto_result["source_language"] != "en":
        await status.edit(content=_same_lang_msg(auto_result["source_language"], passthrough_extra))
        return

    parts = [_fmt_result(auto_result, auto_label, to_lang)]

    if from_lang:
        hint_result = await asyncio.to_thread(translate_text, text, from_lang, to_lang)
        label = hint_label or f"Hint ({get_language_name(from_lang)})"
        parts.append(_fmt_result(hint_result, label, to_lang))

    if analyze:
        analysis = await asyncio.to_thread(_fmt_analyze, text)
        if analysis:
            parts.append(analysis)

    if truncation_note:
        parts.append(truncation_note)

    final_msg = "\n\n".join(parts)
    if len(final_msg) > 1900:
        final_msg = final_msg[:1900] + "\n_[message truncated]_"
    await status.edit(content=final_msg)

    _collect_text(auto_result, text, to_lang, username, filename=filename)


async def _handle_image(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    channel: discord.abc.Messageable,
    author_name: str,
) -> None:
    """Verify, translate, and respond for an image attachment."""
    # 1. Verify
    if attachment.size > _MAX_IMAGE_BYTES:
        await channel.send(
            f"Image `{attachment.filename}` is too large "
            f"({attachment.size / 1024 / 1024:.1f} MB). "
            f"Maximum is {_MAX_IMAGE_BYTES // 1024 // 1024} MB."
        )
        return

    header = await _fetch_header(attachment.url)
    if header:
        err = _check_content_safety(header, attachment.content_type, attachment.filename)
        if err:
            await channel.send(err)
            return

    # 2. Perform
    if _is_same_language(from_lang, to_lang):
        status = await channel.send("Reading image text...")
        try:
            ocr_text, ocr_conf = await asyncio.to_thread(extract_text_combined, attachment.url)
            extra = (
                f"**Detected text** (OCR {ocr_conf * 100:.0f}%):\n> {ocr_text}"
                if ocr_text else "No text detected in the image."
            )
            await status.edit(content=_same_lang_msg(from_lang, extra))
        except Exception as e:
            logger.exception(f"OCR error: {e}")
            await status.edit(content="Failed to read image text.")
        return

    status = await channel.send("Translating image...")
    try:
        result = await asyncio.to_thread(
            translate_image, attachment.url, None, from_lang, to_lang,
            attachment.filename, author_name,
        )
        auto = result["auto"]
        hint = result["hint"]
        collected = result.get("collected_path")

        if collected:
            logger.info(
                "Collected: file=%s | user=%s | attachment=%s | lang=%s | lang_conf=%s | ocr_conf=%s",
                collected, author_name, attachment.filename, auto["source_language"],
                f"{auto['confidence']*100:.0f}%" if auto.get("confidence") is not None else "n/a",
                f"{auto['ocr_confidence']*100:.0f}%" if auto.get("ocr_confidence") is not None else "n/a",
            )
        else:
            logger.debug("Skipped collection (duplicate or no text): user=%s attachment=%s",
                         author_name, attachment.filename)

        # 3. Return result
        if auto["method"] == "none":
            await status.edit(content="No text detected in the image.")
        else:
            parts = [_fmt_result(auto, "Auto", to_lang, ocr=True)]
            if hint:
                from_lang_name = get_language_name(from_lang)
                if hint["method"] != "none":
                    parts.append(_fmt_result(hint, f"Hint ({from_lang_name})", to_lang, ocr=True))
                else:
                    parts.append(f"**Hint ({from_lang_name})** No text detected with hinted reader.")
            if analyze and auto["original_text"]:
                analysis = await asyncio.to_thread(_fmt_analyze, auto["original_text"])
                if analysis:
                    parts.append(analysis)
            await status.edit(content="\n\n".join(parts))
    except Exception as e:
        logger.exception(f"Image translation error: {e}")
        await status.edit(content="Image translation failed. Please try again later.")


async def _handle_audio(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    channel: discord.abc.Messageable,
    author_name: str,
) -> None:
    """Verify, transcribe, translate, and respond for an audio attachment."""
    # 1. Verify
    if attachment.size > _MAX_AUDIO_BYTES:
        await channel.send(
            f"Audio file `{attachment.filename}` is too large "
            f"({attachment.size / 1024 / 1024:.1f} MB). "
            f"Maximum is {_MAX_AUDIO_BYTES // 1024 // 1024} MB."
        )
        return

    header = await _fetch_header(attachment.url)
    if header:
        err = _check_content_safety(header, attachment.content_type, attachment.filename)
        if err:
            await channel.send(err)
            return

    # 2. Perform
    status = await channel.send(f"Transcribing `{attachment.filename}`...")
    try:
        result = await asyncio.to_thread(
            translate_audio, attachment.url, from_lang, to_lang,
            attachment.filename, author_name,
        )

        if not result["original_text"]:
            await status.edit(content="No speech detected in the audio.")
            return

        if result["collected"]:
            logger.info("Collected audio: user=%s | file=%s | lang=%s | method=%s",
                        author_name, attachment.filename,
                        result["source_language"], result["method"])
        else:
            logger.debug("Skipped audio collection (duplicate or no speech): user=%s file=%s",
                         author_name, attachment.filename)

        # 3. Return result — reuses _fmt_result with ocr=True to show transcript + translation
        parts = [_fmt_result(result, "Auto", to_lang, ocr=True)]

        if analyze and result["original_text"]:
            analysis = await asyncio.to_thread(_fmt_analyze, result["original_text"])
            if analysis:
                parts.append(analysis)

        final_msg = "\n\n".join(parts)
        if len(final_msg) > 1900:
            final_msg = final_msg[:1900] + "\n_[message truncated]_"
        await status.edit(content=final_msg)

    except Exception as e:
        logger.exception(f"Audio translation error: {e}")
        await status.edit(content="Audio transcription failed. Please try again later.")


async def _handle_video(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    channel: discord.abc.Messageable,
    author_name: str,
) -> None:
    """Verify, extract audio, transcribe, translate, and respond for a video attachment."""
    # 1. Verify
    if attachment.size > _MAX_VIDEO_BYTES:
        await channel.send(
            f"Video file `{attachment.filename}` is too large "
            f"({attachment.size / 1024 / 1024:.1f} MB). "
            f"Maximum is {_MAX_VIDEO_BYTES // 1024 // 1024} MB."
        )
        return

    header = await _fetch_header(attachment.url)
    if header:
        err = _check_content_safety(header, attachment.content_type, attachment.filename)
        if err:
            await channel.send(err)
            return

    # 2. Perform
    status = await channel.send(f"Extracting audio from `{attachment.filename}`...")
    try:
        result = await asyncio.to_thread(
            translate_video, attachment.url, from_lang, to_lang,
            attachment.filename, author_name,
        )

        if not result["original_text"]:
            await status.edit(content="No speech detected in the video.")
            return

        if result["collected"]:
            logger.info("Collected video: user=%s | file=%s | lang=%s | method=%s",
                        author_name, attachment.filename,
                        result["source_language"], result["method"])
        else:
            logger.debug("Skipped video collection (duplicate or no speech): user=%s file=%s",
                         author_name, attachment.filename)

        # 3. Return result — reuses _fmt_result with ocr=True to show transcript + translation
        parts = [_fmt_result(result, "Auto", to_lang, ocr=True)]

        if analyze and result["original_text"]:
            analysis = await asyncio.to_thread(_fmt_analyze, result["original_text"])
            if analysis:
                parts.append(analysis)

        final_msg = "\n\n".join(parts)
        if len(final_msg) > 1900:
            final_msg = final_msg[:1900] + "\n_[message truncated]_"
        await status.edit(content=final_msg)

    except Exception as e:
        logger.exception(f"Video translation error: {e}")
        await status.edit(content="Video translation failed. Please try again later.")


async def _handle_text_file(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    channel: discord.abc.Messageable,
    author_name: str,
) -> None:
    """Verify, translate, and respond for a plain-text file attachment."""
    # 1. Verify
    if attachment.size > _MAX_TEXT_BYTES:
        await channel.send(
            f"Text file `{attachment.filename}` is too large "
            f"({attachment.size / 1024:.0f} KB). "
            f"Maximum for translation is {_MAX_TEXT_BYTES // 1024} KB."
        )
        return

    status = await channel.send(f"Translating `{attachment.filename}`...")
    try:
        raw = await attachment.read()

        err = _check_content_safety(raw, attachment.content_type, attachment.filename)
        if err:
            await status.edit(content=err)
            return

        try:
            file_text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                file_text = raw.decode("latin-1")
            except UnicodeDecodeError:
                await status.edit(content=f"Could not decode `{attachment.filename}` as text.")
                return

        file_text = file_text.strip()
        if not file_text:
            await status.edit(content=f"`{attachment.filename}` is empty.")
            return

        truncated = len(file_text) > _MAX_TRANSLATE_CHARS
        if truncated:
            file_text = file_text[:_MAX_TRANSLATE_CHARS]

        if _is_same_language(from_lang, to_lang):
            await status.edit(content=_same_lang_msg(from_lang))
            return

        # 2. Perform + 3. Return result
        hint_label = (
            f"Hint ({get_language_name(from_lang)}) (`{attachment.filename}`)"
            if from_lang else None
        )
        await _run_text_translate(
            file_text, from_lang, to_lang, analyze, status,
            username=author_name,
            auto_label=f"Auto (`{attachment.filename}`)",
            hint_label=hint_label,
            truncation_note=(
                f"_Note: file truncated to {_MAX_TRANSLATE_CHARS} characters for translation._"
                if truncated else ""
            ),
            filename=attachment.filename,
        )

    except discord.HTTPException as e:
        logger.exception(f"Failed to download text attachment: {e}")
        await status.edit(content=f"Could not download `{attachment.filename}`. Please try again.")
    except Exception as e:
        logger.exception(f"Text file translation error: {e}")
        await status.edit(content="Text file translation failed. Please try again later.")


async def _handle_text_inline(
    text: str,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    channel: discord.abc.Messageable,
    author_name: str,
) -> None:
    """Verify, translate, and respond for inline /translate text."""
    # 1. Verify
    if not text:
        await channel.send("Please provide text after `/translate`.")
        return
    if _is_same_language(from_lang, to_lang):
        await channel.send(_same_lang_msg(from_lang))
        return

    # 2. Perform + 3. Return result
    status = await channel.send("Translating...")
    try:
        await _run_text_translate(
            text, from_lang, to_lang, analyze, status,
            username=author_name,
            passthrough_extra=f"**Detected text**:\n> {text}",
        )
    except Exception as e:
        logger.exception(f"Translation error: {e}")
        await status.edit(content="Translation failed. Please try again later.")


# Bot events
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = " ".join(message.content.split()[1:])
    attachments = ", ".join(a.filename for a in message.attachments)
    attachment_str = f" | attachments: {attachments}" if attachments else ""
    print(f"Received message from [{message.author.name}]: {msg}{attachment_str}")

    if not (client.user.mentioned_in(message) and message.content.startswith(client.user.mention)):
        return

    if not msg:
        await message.channel.send(
            "Hello! I am TL-Bot. Please send me an image with text or include text when @-ing me and I will do my best to translate it for you!"
        )
        return

    if msg.startswith("/help"):
        await message.channel.send(
            "Available commands:\n"
            "/help — Show this help message\n"
            "/translate — Translate text or image\n"
            "  --from <language>  Hint the source language (also shows auto-detect result)\n"
            "  --to <language>    Set the output language (default: English)\n"
            "  --analyze          Show detected language segments alongside the translation"
        )
        return

    if msg.startswith("/translate"):
        cmd = msg[len("/translate"):].strip()
        text_input, from_lang, to_lang, analyze, errors = _parse_translate_flags(cmd)
        for err in errors:
            await message.channel.send(err)

        if message.attachments:
            for attachment in message.attachments:
                if "image" in attachment.content_type:
                    await _handle_image(attachment, from_lang, to_lang, analyze, message.channel, message.author.name)
                elif "audio" in attachment.content_type:
                    await _handle_audio(attachment, from_lang, to_lang, analyze, message.channel, message.author.name)
                elif "video" in attachment.content_type:
                    await _handle_video(attachment, from_lang, to_lang, analyze, message.channel, message.author.name)
                elif "text" in attachment.content_type:
                    await _handle_text_file(attachment, from_lang, to_lang, analyze, message.channel, message.author.name)
                else:
                    await message.channel.send(f"Unsupported file type: {attachment.url}.")
        else:
            await _handle_text_inline(text_input, from_lang, to_lang, analyze, message.channel, message.author.name)
        return

    await message.channel.send("Translate function not yet implemented, please stay tuned!")


# Token loaded from DISCORD_BOT_TOKEN in .env or environment
bot_token = os.getenv("DISCORD_BOT_TOKEN")
if not bot_token:
    raise RuntimeError("DISCORD_BOT_TOKEN is not set. Add it to your .env file.")

client.run(bot_token, log_handler=None)
