# TL-Bot.py
# Discord bot for handling image/text translations

import asyncio
import io
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
from ocr import extract_text_combined, extract_text  # noqa: E402
from synthesize_image import synthesize_image, synthesize_text_to_image  # noqa: E402

# Add audio translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "3-Audio"))
from translate_audio import translate_audio  # noqa: E402
from synthesize_audio import synthesize as synthesize_speech  # noqa: E402

# Add video translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "4-Video"))
from translate_video import translate_video  # noqa: E402
from synthesize_video import synthesize_video  # noqa: E402

# Text collection
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "0-Data" / "Text" / "training"))
from collect_text import save_submission as save_text_submission  # noqa: E402

# Synthesized output collection
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "0-Data" / "Synthesized"))
from collect_synthesized import save_synthesis as save_synthesis_output  # noqa: E402

# Prompt chat module
sys.path.insert(0, str(Path(__file__).parent / "Prompt"))
from prompt import ask as prompt_ask  # noqa: E402

# Logging rotation settings
_LOG_MAX_BYTES    = 32 * 1024 * 1024   # 32 MiB per log file
_LOG_BACKUP_COUNT = 5                  # number of rotated files to retain

# Discord message length cap; actual limit is 2000 but we leave a 100-char
# safety margin for the truncation notice appended below.
_DISCORD_MSG_LIMIT = 1900

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
    maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT,
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
_MAX_PROMPT_CHARS   = 2000               # user input cap for /prompt
_MAX_HISTORY_TURNS  = 10                 # rolling window: 10 user/assistant pairs

# Per-user conversation history for /prompt; keyed by Discord user ID.
# Ephemeral — cleared on bot restart.
_prompt_history: dict[int, list[dict]] = {}


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


_SYNTHESIZE_TYPES = ("audio", "image", "text", "video")


def _parse_translate_flags(cmd: str) -> tuple[str, str | None, str | None, bool, str | None, list[str]]:
    """Parse --from, --to, --analyze, and --synthesize flags out of a /translate command string.

    Returns (remaining_text, from_lang, to_lang, analyze, synthesize_type, error_messages).
    synthesize_type is one of 'audio', 'image', 'text', or None if the flag was not given.
    remaining_text is the command string with all recognized flags removed.
    """
    tokens = cmd.split()
    from_lang = to_lang = None
    analyze = False
    synthesize: str | None = None
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
        elif tokens[i] == "--synthesize":
            if i + 1 < len(tokens) and tokens[i + 1] in _SYNTHESIZE_TYPES:
                synthesize = tokens[i + 1]
                i += 2
            else:
                errors.append(
                    f"`--synthesize` requires a type: {', '.join(f'`{t}`' for t in _SYNTHESIZE_TYPES)}."
                )
                i += 1
        else:
            remaining.append(tokens[i])
            i += 1
    return " ".join(remaining), from_lang, to_lang, analyze, synthesize, errors


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


async def _send_synthesized(
    channel: discord.abc.Messageable,
    translated_text: str,
    synthesize_type: str,
    to_lang: str | None,
    *,
    username: str,
    source_type: str,
) -> None:
    """Send a synthesized output file for 'audio' or 'text' synthesis types.

    'audio' → MP3 via gTTS.  'text' → UTF-8 .txt file.
    'image' and 'video' are handled per-handler because they require source media.
    Saves the output bytes to 0-Data/Synthesized/<type>/data/ (non-fatal).
    """
    if synthesize_type == "audio":
        try:
            audio_bytes = await asyncio.to_thread(
                synthesize_speech, translated_text, to_lang or "en"
            )
            await channel.send(
                "**Synthesized translation:**",
                file=discord.File(io.BytesIO(audio_bytes), filename="translated.mp3"),
            )
            try:
                save_synthesis_output(
                    audio_bytes, "audio",
                    translated_text=translated_text,
                    source_type=source_type,
                    target_language=to_lang or "en",
                    username=username,
                )
            except Exception:
                logger.warning("Failed to save audio synthesis output", exc_info=True)
        except Exception as e:
            logger.warning("Audio synthesis failed: %s", e)
            await channel.send("_Audio synthesis failed._")
    elif synthesize_type == "text":
        txt_bytes = translated_text.encode("utf-8")
        await channel.send(
            "**Synthesized translation:**",
            file=discord.File(io.BytesIO(txt_bytes), filename="translated.txt"),
        )
        try:
            save_synthesis_output(
                txt_bytes, "text",
                translated_text=translated_text,
                source_type=source_type,
                target_language=to_lang or "en",
                username=username,
            )
        except Exception:
            logger.warning("Failed to save text synthesis output", exc_info=True)


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
    synthesize: str | None = None,
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
    if len(final_msg) > _DISCORD_MSG_LIMIT:
        final_msg = final_msg[:_DISCORD_MSG_LIMIT] + "\n_[message truncated]_"
    await status.edit(content=final_msg)

    _collect_text(auto_result, text, to_lang, username, filename=filename)

    if synthesize and auto_result.get("translated_text"):
        if synthesize == "image":
            try:
                synth_bytes = await asyncio.to_thread(
                    synthesize_text_to_image, auto_result["translated_text"], to_lang or "en"
                )
                await status.channel.send(
                    "**Synthesized translation:**",
                    file=discord.File(io.BytesIO(synth_bytes), filename="translated.png"),
                )
                try:
                    save_synthesis_output(
                        synth_bytes, "image",
                        translated_text=auto_result["translated_text"],
                        source_type="text",
                        target_language=to_lang or "en",
                        username=username,
                    )
                except Exception:
                    logger.warning("Failed to save image synthesis output", exc_info=True)
            except Exception as e:
                logger.warning("Image synthesis failed: %s", e)
                await status.channel.send("_Image synthesis failed._")
        elif synthesize == "video":
            await status.channel.send("_Video synthesis is only supported for video input._")
        else:
            await _send_synthesized(status.channel, auto_result["translated_text"], synthesize, to_lang,
                                    username=username, source_type="text")


async def _handle_image(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    synthesize: str | None,
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

            if synthesize and auto["translated_text"] and auto["method"] != "none":
                if synthesize == "image":
                    try:
                        segments = await asyncio.to_thread(extract_text, attachment.url)
                        if segments:
                            synth_bytes = await asyncio.to_thread(
                                synthesize_image, attachment.url, segments,
                                auto["translated_text"], to_lang or "en",
                            )
                            await channel.send(
                                "**Synthesized translation:**",
                                file=discord.File(io.BytesIO(synth_bytes), filename="translated.png"),
                            )
                            try:
                                save_synthesis_output(
                                    synth_bytes, "image",
                                    translated_text=auto["translated_text"],
                                    source_type="image",
                                    target_language=to_lang or "en",
                                    username=author_name,
                                )
                            except Exception:
                                logger.warning("Failed to save image synthesis output", exc_info=True)
                        else:
                            await channel.send("_No text regions found for image synthesis._")
                    except Exception as e:
                        logger.warning("Image synthesis failed: %s", e)
                        await channel.send("_Image synthesis failed._")
                elif synthesize == "video":
                    await channel.send("_Video synthesis is only supported for video input._")
                else:
                    await _send_synthesized(channel, auto["translated_text"], synthesize, to_lang,
                                            username=author_name, source_type="image")

    except Exception as e:
        logger.exception(f"Image translation error: {e}")
        await status.edit(content="Image translation failed. Please try again later.")


async def _handle_audio(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    synthesize: str | None,
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
        if len(final_msg) > _DISCORD_MSG_LIMIT:
            final_msg = final_msg[:_DISCORD_MSG_LIMIT] + "\n_[message truncated]_"
        await status.edit(content=final_msg)

        if synthesize and result["translated_text"]:
            if synthesize == "image":
                try:
                    synth_bytes = await asyncio.to_thread(
                        synthesize_text_to_image, result["translated_text"], to_lang or "en"
                    )
                    await channel.send(
                        "**Synthesized translation:**",
                        file=discord.File(io.BytesIO(synth_bytes), filename="translated.png"),
                    )
                    try:
                        save_synthesis_output(
                            synth_bytes, "image",
                            translated_text=result["translated_text"],
                            source_type="audio",
                            target_language=to_lang or "en",
                            username=author_name,
                        )
                    except Exception:
                        logger.warning("Failed to save image synthesis output", exc_info=True)
                except Exception as e:
                    logger.warning("Image synthesis failed: %s", e)
                    await channel.send("_Image synthesis failed._")
            elif synthesize == "video":
                await channel.send("_Video synthesis is only supported for video input._")
            else:
                await _send_synthesized(channel, result["translated_text"], synthesize, to_lang,
                                        username=author_name, source_type="audio")

    except Exception as e:
        logger.exception(f"Audio translation error: {e}")
        await status.edit(content="Audio transcription failed. Please try again later.")


async def _handle_video(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    synthesize: str | None,
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
        if len(final_msg) > _DISCORD_MSG_LIMIT:
            final_msg = final_msg[:_DISCORD_MSG_LIMIT] + "\n_[message truncated]_"
        await status.edit(content=final_msg)

        if synthesize and result["translated_text"]:
            if synthesize == "image":
                try:
                    synth_bytes = await asyncio.to_thread(
                        synthesize_text_to_image, result["translated_text"], to_lang or "en"
                    )
                    await channel.send(
                        "**Synthesized translation:**",
                        file=discord.File(io.BytesIO(synth_bytes), filename="translated.png"),
                    )
                    try:
                        save_synthesis_output(
                            synth_bytes, "image",
                            translated_text=result["translated_text"],
                            source_type="video",
                            target_language=to_lang or "en",
                            username=author_name,
                        )
                    except Exception:
                        logger.warning("Failed to save image synthesis output", exc_info=True)
                except Exception as e:
                    logger.warning("Image synthesis failed: %s", e)
                    await channel.send("_Image synthesis failed._")
            elif synthesize == "video":
                try:
                    await status.edit(content=final_msg + "\n_Synthesizing translated video..._")
                    video_bytes = await asyncio.to_thread(
                        synthesize_video, attachment.url, result["translated_text"], to_lang or "en"
                    )
                    await channel.send(
                        "**Synthesized translation:**",
                        file=discord.File(io.BytesIO(video_bytes), filename="translated.mkv"),
                    )
                    await status.edit(content=final_msg)
                    try:
                        save_synthesis_output(
                            video_bytes, "video",
                            translated_text=result["translated_text"],
                            source_type="video",
                            target_language=to_lang or "en",
                            username=author_name,
                        )
                    except Exception:
                        logger.warning("Failed to save video synthesis output", exc_info=True)
                except Exception as e:
                    logger.warning("Video synthesis failed: %s", e)
                    await channel.send("_Video synthesis failed._")
                    await status.edit(content=final_msg)
            else:
                await _send_synthesized(channel, result["translated_text"], synthesize, to_lang,
                                        username=author_name, source_type="video")

    except Exception as e:
        logger.exception(f"Video translation error: {e}")
        await status.edit(content="Video translation failed. Please try again later.")


async def _handle_text_file(
    attachment: discord.Attachment,
    from_lang: str | None,
    to_lang: str | None,
    analyze: bool,
    synthesize: str | None,
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
            synthesize=synthesize,
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
    synthesize: str | None,
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
            synthesize=synthesize,
        )
    except Exception as e:
        logger.exception(f"Translation error: {e}")
        await status.edit(content="Translation failed. Please try again later.")


# Per-language test scripts for /test <lang>.
# Each entry: (input_text, to_lang) — to_lang is None to target English,
# except "en" which targets Spanish so there is something to actually translate.
_TEST_SCRIPTS: dict[str, tuple[str, str | None]] = {
    "english": (
        "Hello! I am TL-Bot, a translation assistant for Discord. "
        "I can translate text, images, audio files, and videos from many languages. "
        "For example, if you send me an image containing Chinese or Japanese text, "
        "I will automatically read the text and translate it for you. "
        "You can also send voice messages, which I will transcribe using Whisper and then translate. "
        "If you would like the result as an audio file, a plain text file, or an image, "
        "just add the --synthesize flag to your translate command. "
        "Type /help for a full list of commands and options!",
        "es",
    ),
    "chinese": (
        "你好！我是TL-Bot，一个Discord翻译助手。"
        "我可以翻译文字、图片、音频文件和视频中的内容。"
        "例如，如果你发送一张包含中文或日文的图片，我会自动识别文字并将其翻译成英文。"
        "你也可以发送语音消息，我会使用Whisper进行转录，然后进行翻译。"
        "如果你希望以音频文件、纯文本文件或图片的形式接收翻译结果，"
        "只需在翻译命令中添加 --synthesize 标志即可。"
        "输入 /help 查看完整的命令和选项列表！",
        None,
    ),
    "japanese": (
        "こんにちは！私はDiscord用の翻訳アシスタント、TL-Botです。"
        "テキスト、画像、音声ファイル、動画など、さまざまな形式のコンテンツを翻訳できます。"
        "例えば、中国語や日本語のテキストが含まれた画像を送っていただければ、"
        "自動的にテキストを認識して英語に翻訳します。"
        "音声メッセージも受け付けており、Whisperで文字起こしをした後に翻訳します。"
        "翻訳結果を音声ファイル、テキストファイル、または画像として受け取りたい場合は、"
        "翻訳コマンドに --synthesize フラグを追加してください。"
        "/help と入力すると、コマンドとオプションの一覧が表示されます！",
        None,
    ),
    "korean": (
        "안녕하세요! 저는 TL-Bot입니다. "
        "저는 이미지, 텍스트, 오디오 파일을 번역할 수 있어요. "
        "예를 들어, 한국어나 중국어로 된 이미지를 보내주시면 자동으로 텍스트를 인식하고 영어로 번역해드립니다. "
        "또한 음성 메시지도 받아서 Whisper로 전사한 뒤 번역할 수 있습니다. "
        "번역 결과는 텍스트, 오디오 파일(MP3), 또는 이미지 형태로 받아보실 수 있어요. "
        "궁금한 점이 있으시면 /help 명령어를 사용해 주세요!",
        None,
    ),
    "french": (
        "Bonjour ! Je suis TL-Bot, un assistant de traduction pour Discord. "
        "Je peux traduire du texte, des images, des fichiers audio et des vidéos depuis de nombreuses langues. "
        "Par exemple, si vous m'envoyez une image contenant du texte en chinois ou en japonais, "
        "je lirai automatiquement le texte et le traduirai pour vous. "
        "Vous pouvez également envoyer des messages vocaux, que je transcrirai avec Whisper avant de les traduire. "
        "Si vous souhaitez recevoir le résultat sous forme de fichier audio, de texte brut ou d'image, "
        "ajoutez simplement le drapeau --synthesize à votre commande de traduction. "
        "Tapez /help pour obtenir la liste complète des commandes et des options !",
        None,
    ),
}

_TEST_LANG_ALIASES: dict[str, str] = {
    "en": "english", "zh": "chinese", "cn": "chinese", "ja": "japanese",
    "jp": "japanese", "ko": "korean", "kr": "korean", "fr": "french",
}


async def _handle_test(channel: discord.abc.Messageable, author_name: str, lang_arg: str) -> None:
    """Translate one of the built-in test scripts and return all three synthesis outputs.

    lang_arg is the word after /test (e.g. "korean", "zh"). Defaults to "korean".
    Video synthesis is omitted — it requires a video attachment as source.
    """
    key = _TEST_LANG_ALIASES.get(lang_arg, lang_arg) if lang_arg else "korean"
    if key not in _TEST_SCRIPTS:
        valid = ", ".join(f"`{k}`" for k in _TEST_SCRIPTS)
        await channel.send(
            f"Unknown test language `{lang_arg}`. Valid options: {valid}."
        )
        return

    test_text, to_lang = _TEST_SCRIPTS[key]
    tgt_label = get_language_name(to_lang) if to_lang else "English"

    status = await channel.send(
        f"**Running /test {key}**\nInput ({len(test_text)} chars):\n> {test_text}\n\n_Translating..._"
    )
    try:
        result = await asyncio.to_thread(translate_text, test_text, None, to_lang)
    except Exception as e:
        logger.exception("Test translation failed: %s", e)
        await status.edit(content="_Translation failed during /test._")
        return

    translated = result.get("translated_text", "")
    src_name = get_language_name(result.get("source_language", ""))
    conf = result.get("confidence")
    conf_str = f" ({conf * 100:.0f}%)" if conf is not None else ""
    synth_lang = to_lang or "en"

    await status.edit(
        content=(
            f"**[/test {key}] Translation result [{src_name} → {tgt_label}]{conf_str}**\n"
            f"> {test_text}\n"
            f"{translated}\n\n"
            "_Generating synthesized outputs..._"
        )
    )

    if not translated:
        await status.edit(content="_No translated text produced — synthesis skipped._")
        return

    # Generate all three outputs; collect whichever succeed.
    synth_files: list[discord.File] = []
    synth_errors: list[str] = []

    # --- text file ---
    try:
        txt_bytes = translated.encode("utf-8")
        synth_files.append(discord.File(io.BytesIO(txt_bytes), filename=f"test_{key}_translated.txt"))
        try:
            save_synthesis_output(
                txt_bytes, "text",
                translated_text=translated,
                source_type="text",
                target_language=synth_lang,
                username=author_name,
            )
        except Exception:
            logger.warning("Test: failed to save text synthesis output", exc_info=True)
    except Exception as e:
        logger.warning("Test text-file synthesis failed: %s", e)
        synth_errors.append("text file")

    # --- audio MP3 ---
    try:
        audio_bytes = await asyncio.to_thread(synthesize_speech, translated, synth_lang)
        synth_files.append(discord.File(io.BytesIO(audio_bytes), filename=f"test_{key}_translated.mp3"))
        try:
            save_synthesis_output(
                audio_bytes, "audio",
                translated_text=translated,
                source_type="text",
                target_language=synth_lang,
                username=author_name,
            )
        except Exception:
            logger.warning("Test: failed to save audio synthesis output", exc_info=True)
    except Exception as e:
        logger.warning("Test audio synthesis failed: %s", e)
        synth_errors.append("audio")

    # --- image PNG ---
    try:
        synth_bytes = await asyncio.to_thread(synthesize_text_to_image, translated, synth_lang)
        synth_files.append(discord.File(io.BytesIO(synth_bytes), filename=f"test_{key}_translated.png"))
        try:
            save_synthesis_output(
                synth_bytes, "image",
                translated_text=translated,
                source_type="text",
                target_language=synth_lang,
                username=author_name,
            )
        except Exception:
            logger.warning("Test: failed to save image synthesis output", exc_info=True)
    except Exception as e:
        logger.warning("Test image synthesis failed: %s", e)
        synth_errors.append("image")

    error_note = f"\n_Failed: {', '.join(synth_errors)}_" if synth_errors else ""
    await status.edit(
        content=(
            f"**[/test {key}] Translation result [{src_name} → {tgt_label}]{conf_str}**\n"
            f"> {test_text}\n"
            f"{translated}{error_note}"
        )
    )
    if synth_files:
        await channel.send(
            f"**[/test {key}] Synthesized outputs (.txt · .mp3 · .png)**",
            files=synth_files,
        )


async def _handle_prompt(text: str, channel: discord.abc.Messageable, author_id: int) -> None:
    """Send user text to the chat model and reply, maintaining per-user history."""
    if not text.strip():
        await channel.send("_Usage: `/prompt <message>`_")
        return

    text = text[:_MAX_PROMPT_CHARS]
    history = _prompt_history.setdefault(author_id, [])
    history.append({"role": "user", "content": text})

    # Trim to rolling window
    max_messages = _MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        del history[:len(history) - max_messages]

    status = await channel.send("_Thinking..._")

    try:
        reply = await asyncio.to_thread(prompt_ask, list(history))
    except Exception as e:
        logger.exception("Prompt failed: %s", e)
        history.pop()
        await status.edit(content="_Prompt request failed. Please try again._")
        return

    history.append({"role": "assistant", "content": reply})

    if len(reply) > _DISCORD_MSG_LIMIT:
        reply = reply[:_DISCORD_MSG_LIMIT] + "\n_(response truncated)_"

    await status.edit(content=reply)


async def _handle_history(channel: discord.abc.Messageable, author_id: int) -> None:
    """Display the calling user's conversation history."""
    history = _prompt_history.get(author_id, [])
    if not history:
        await channel.send("_No conversation history yet. Start with `/prompt <message>`._")
        return

    lines = ["**Conversation history:**"]
    turn = 0
    for msg in history:
        if msg["role"] == "user":
            turn += 1
            label = f"[{turn}] **You:** "
        else:
            label = f"[{turn}] **Bot:** "
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"{label}{content}")

    output = "\n".join(lines)
    if len(output) > _DISCORD_MSG_LIMIT:
        output = output[:_DISCORD_MSG_LIMIT] + "\n_(truncated)_"

    await channel.send(output)


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
            "  --analyze          Show detected language segments alongside the translation\n"
            "  --synthesize <type>  Return a synthesized output file of the translation\n"
            "      audio  → MP3 speech file (all input types)\n"
            "      text   → plain .txt file (all input types)\n"
            "      image  → translated PNG (image: text replaced in-place; other inputs: text on plain background)\n"
            "      video  → MKV with original video and synthesized translated audio (video input only)\n"
            "/test <language> — Run a self-test with a built-in sample text and return all synthesis outputs\n"
            "      Supported: english, chinese, japanese, korean, french\n"
            "/prompt <message> — Ask the bot a question (conversation history is maintained per user)\n"
            "/history — Show your conversation history"
        )
        return

    if msg.startswith("/translate"):
        cmd = msg[len("/translate"):].strip()
        text_input, from_lang, to_lang, analyze, synthesize, errors = _parse_translate_flags(cmd)
        for err in errors:
            await message.channel.send(err)

        if message.attachments:
            for attachment in message.attachments:
                if "image" in attachment.content_type:
                    await _handle_image(attachment, from_lang, to_lang, analyze, synthesize, message.channel, message.author.name)
                elif "audio" in attachment.content_type:
                    await _handle_audio(attachment, from_lang, to_lang, analyze, synthesize, message.channel, message.author.name)
                elif "video" in attachment.content_type:
                    await _handle_video(attachment, from_lang, to_lang, analyze, synthesize, message.channel, message.author.name)
                elif "text" in attachment.content_type:
                    await _handle_text_file(attachment, from_lang, to_lang, analyze, synthesize, message.channel, message.author.name)
                else:
                    await message.channel.send(f"Unsupported file type: {attachment.url}.")
        else:
            await _handle_text_inline(text_input, from_lang, to_lang, analyze, synthesize, message.channel, message.author.name)
        return

    if msg.startswith("/test"):
        lang_arg = msg[len("/test"):].strip().lower()
        await _handle_test(message.channel, message.author.name, lang_arg)
        return

    if msg.startswith("/prompt"):
        prompt_text = msg[len("/prompt"):].strip()
        await _handle_prompt(prompt_text, message.channel, message.author.id)
        return

    if msg.startswith("/history"):
        await _handle_history(message.channel, message.author.id)
        return

    await message.channel.send("Unrecognized command. Type `/help` for a list of available commands.")


# Token loaded from DISCORD_BOT_TOKEN in .env or environment
bot_token = os.getenv("DISCORD_BOT_TOKEN")
if not bot_token:
    raise RuntimeError("DISCORD_BOT_TOKEN is not set. Add it to your .env file.")

client.run(bot_token, log_handler=None)
