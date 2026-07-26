# TL-Bot.py
# Discord bot for handling image/text translations

import asyncio
import datetime
import io
import os
import re
import sys
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

# User recognition / chat history collection
sys.path.insert(0, str(Path(__file__).parent / "UserRecognition" / "0-Data" / "training"))
import collect_history as _collect_history  # noqa: E402

# User recognition inference
sys.path.insert(0, str(Path(__file__).parent / "UserRecognition"))
import identify as _identify  # noqa: E402

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
intents.members = True  # required for guild.chunk() to return full member list

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


_COLLECT_MENTION_RE = re.compile(r'<@!?(\d+)>')
_COLLECT_CHANNEL_RE = re.compile(r'<#(\d+)>')

# A quoted span, or a run of non-whitespace. Lets display names containing
# spaces be passed as one target: /collect "Paul ohannigan"
_COLLECT_TOKEN_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'|(\S+)')


def _tokenize_collect(cmd: str) -> list:
    """Split a /collect argument string into [(token, was_quoted), ...].

    Quoted tokens are never treated as flags and never split, so a name that
    collides with a flag or contains spaces still resolves.
    """
    out = []
    for m in _COLLECT_TOKEN_RE.finditer(cmd):
        dq, sq, bare = m.groups()
        if dq is not None:
            if dq.strip():
                out.append((dq.strip(), True))
        elif sq is not None:
            if sq.strip():
                out.append((sq.strip(), True))
        else:
            out.append((bare, False))
    return out


def _parse_collect_flags(cmd: str, guild, require_targets: bool = True) -> tuple:
    """Parse /collect (and /index) arguments.

    Returns (targets, channel, since, limit, errors).
    targets is a list of int user IDs and/or str names (one per space-separated token),
    or ["__BATCH__"] when --batch is given.
    channel is a discord.TextChannel or None (means all channels).
    since is a datetime.datetime (UTC) or None.
    limit is an int or None (means no limit).

    require_targets=False is used by /index, which takes the same --channel /
    --since / --limit flags but no user targets.

    Batch mode: --batch reads targets from data/{guild_id}/targets.txt (one username/ID per line).
    Space-separated: /collect user1 user2 @mention3 collects all three in sequence.
    Quoted: /collect "Paul ohannigan" passes a display name containing spaces as one target.
    """
    errors: list[str] = []
    targets: list = []
    channel = None
    since: datetime.datetime | None = None
    limit: int | None = None
    batch = False

    parsed = _tokenize_collect(cmd)
    tokens = [t for t, _ in parsed]
    quoted = [q for _, q in parsed]
    remaining: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_flag = not quoted[i]
        if is_flag and tok == "--batch":
            batch = True
        elif is_flag and tok == "--channel" and i + 1 < len(tokens):
            i += 1
            val = tokens[i]
            m = _COLLECT_CHANNEL_RE.match(val)
            cid = int(m.group(1)) if m else (int(val) if val.isdigit() else None)
            if cid is None:
                errors.append(f"_Invalid `--channel` value: `{val}`_")
            elif guild:
                ch = guild.get_channel(cid)
                if ch is None:
                    errors.append(f"_Channel `{cid}` not found in this server._")
                else:
                    channel = ch
        elif is_flag and tok == "--since" and i + 1 < len(tokens):
            i += 1
            try:
                since = datetime.datetime.strptime(tokens[i], "%Y-%m-%d").replace(
                    tzinfo=datetime.timezone.utc
                )
            except ValueError:
                errors.append(f"_Invalid `--since` date `{tokens[i]}` — use YYYY-MM-DD._")
        elif is_flag and tok == "--limit" and i + 1 < len(tokens):
            i += 1
            if tokens[i].isdigit():
                limit = int(tokens[i])
            else:
                errors.append(f"_Invalid `--limit` value: `{tokens[i]}`_")
        else:
            remaining.append(tok)
        i += 1

    if batch:
        targets = ["__BATCH__"]
    elif remaining:
        for tok in remaining:
            m = _COLLECT_MENTION_RE.match(tok)
            if m:
                targets.append(int(m.group(1)))
            elif tok.isdigit():
                targets.append(int(tok))
            else:
                targets.append(tok)  # plain name — resolved against identity table
    elif require_targets:
        errors.append(
            "_Usage: `/collect user1 [user2 ...] [--channel #ch] [--since YYYY-MM-DD] [--limit N]`_\n"
            "_Quote names containing spaces: `/collect \"Display Name\"`_\n"
            "_`/index` builds the user index (needed to reach users who have left)._\n"
            "_`/collect --batch` collects all users listed in `targets.txt` for this guild._"
        )

    return targets, channel, since, limit, errors


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


async def _scan_members(guild: discord.Guild, guild_id: str) -> int:
    """Index all guild members into identity.jsonl and update guilds.jsonl. Returns member count."""
    if not guild.chunked:
        await guild.chunk()  # request full member list if not yet received
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = [
        {
            "user_id":      str(m.id),
            "guild_id":     guild_id,
            "username":     m.name,
            "display_name": m.display_name,
            "bot":          m.bot,
            "indexed_at":   now_iso,
        }
        for m in guild.members
    ]
    _collect_history.save_guild(guild_id, guild.name, now_iso)
    _collect_history.save_identity(records, guild_id)
    return len(records)


async def _scan_user_channels(
    guild: discord.Guild,
    guild_id: str,
    target_id: int | None,
    channels: list,
    since,
    limit,
    seen_ids: set,
    status: discord.Message,
    label: str = "",
) -> tuple:
    """Scan channels for messages from target_id. Returns (msg_found, msg_new, skipped, first_ts, last_ts).

    Every author encountered is recorded into authors.jsonl, not just the target.
    The loop already visits each message, so indexing the authors costs nothing
    and is what later makes a departed user resolvable by name.

    target_id=None performs an index-only sweep: authors are recorded but no
    messages are stored. `/index` uses this, because the index has to
    be buildable *before* a departed user can be resolved — otherwise resolving
    them would require an index that only a successful resolution could create.
    """
    total_ch  = len(channels)
    msg_found = 0
    msg_new   = 0
    skipped   = []
    first_ts: str | None = None
    last_ts:  str | None = None
    seen_authors: dict = {}

    for idx, ch in enumerate(channels, 1):
        if idx % 5 == 0 or idx == total_ch:
            prefix = f"[{label}] " if label else ""
            await status.edit(content=f"_{prefix}Scanning channel {idx}/{total_ch}..._")

        await asyncio.sleep(0)

        perms = ch.permissions_for(guild.me)
        if not perms.read_messages or not perms.read_message_history:
            skipped.append(ch.name)
            _collect_history.save_channel({
                "channel_id": str(ch.id), "guild_id": guild_id,
                "channel_name": ch.name, "channel_type": str(ch.type),
                "included": False, "skip_reason": "no_permission",
                "message_count_collected": 0,
            }, guild_id)
            continue

        ch_count = 0
        retry_after = 1.0
        while True:
            try:
                after = discord.Object(id=discord.utils.time_snowflake(since)) if since else None
                async for msg in ch.history(limit=limit, after=after, oldest_first=True):
                    # Index every author seen, before filtering to the target.
                    a_id = str(msg.author.id)
                    a_ts = msg.created_at.isoformat()
                    prior = seen_authors.get(a_id)
                    if prior is None:
                        seen_authors[a_id] = {
                            "user_id":       a_id,
                            "guild_id":      guild_id,
                            "username":      msg.author.name,
                            "display_name":  getattr(msg.author, "display_name", msg.author.name),
                            "is_bot":        bool(msg.author.bot),
                            "first_seen_at": a_ts,
                            "last_seen_at":  a_ts,
                        }
                    else:
                        if a_ts < prior["first_seen_at"]:
                            prior["first_seen_at"] = a_ts
                        if a_ts > prior["last_seen_at"]:
                            prior["last_seen_at"] = a_ts

                    # target_id None = index-only sweep: record who posted,
                    # store nothing. Lets the author index be built before any
                    # target has been resolved.
                    if target_id is None or msg.author.id != target_id:
                        continue

                    reply_author_id   = None
                    reply_author_name = None
                    if (
                        msg.reference
                        and isinstance(getattr(msg.reference, "resolved", None), discord.Message)
                    ):
                        reply_author_id   = str(msg.reference.resolved.author.id)
                        reply_author_name = msg.reference.resolved.author.name

                    ts = msg.created_at.isoformat()
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts

                    record = {
                        "message_id":           str(msg.id),
                        "guild_id":             guild_id,
                        "channel_id":           str(ch.id),
                        "channel_name":         ch.name,
                        "author_id":            str(msg.author.id),
                        "author_name":          msg.author.name,
                        "content_raw":          msg.content,
                        "content_normalized":   _collect_history.normalize_content(msg.content),
                        "timestamp":            ts,
                        "edited_timestamp":     msg.edited_at.isoformat() if msg.edited_at else None,
                        "is_reply":             msg.reference is not None,
                        "reply_to_author_id":   reply_author_id,
                        "reply_to_author_name": reply_author_name,
                        "has_attachments":      bool(msg.attachments),
                        "has_embeds":           bool(msg.embeds),
                        "token_count":          len(msg.content.split()),
                    }
                    if _collect_history.save_message(record, guild_id, seen_ids):
                        msg_new += 1
                    msg_found += 1
                    ch_count  += 1
                break

            except discord.Forbidden:
                skipped.append(ch.name)
                _collect_history.save_channel({
                    "channel_id": str(ch.id), "guild_id": guild_id,
                    "channel_name": ch.name, "channel_type": str(ch.type),
                    "included": False, "skip_reason": "no_permission",
                    "message_count_collected": 0,
                }, guild_id)
                break

            except discord.HTTPException as exc:
                if exc.status == 429:
                    wait = float(exc.response.headers.get("Retry-After", retry_after))
                    logger.warning("Rate limited on #%s — waiting %.1fs", ch.name, wait)
                    await status.edit(content=f"_Rate limited — waiting {wait:.0f}s..._")
                    await asyncio.sleep(wait)
                    retry_after = min(retry_after * 2, 60.0)
                else:
                    logger.warning("HTTPException on #%s: %s", ch.name, exc)
                    skipped.append(ch.name)
                    break

        _collect_history.save_channel({
            "channel_id": str(ch.id), "guild_id": guild_id,
            "channel_name": ch.name, "channel_type": str(ch.type),
            "included": ch.name not in skipped, "skip_reason": None,
            "message_count_collected": ch_count,
        }, guild_id)

    if seen_authors:
        try:
            new_authors = _collect_history.save_authors(list(seen_authors.values()), guild_id)
            if new_authors:
                logger.info("Indexed %d new author(s) into authors.jsonl for guild %s",
                            new_authors, guild_id)
        except Exception:
            logger.exception("Failed to save author index for guild %s", guild_id)

    return msg_found, msg_new, skipped, first_ts, last_ts


async def _resolve_collect_target(target_raw, guild: discord.Guild, identity: dict,
                                  member_count: int, authors: dict | None = None):
    """Resolve a collect target (int ID, str name, or @mention str) to a Member or User.

    Users who have left the server are still resolvable: their messages remain in
    channel history, and `_scan_user_channels` filters on a plain user ID rather
    than a Member object. Name lookup falls through identity.jsonl (current
    members) to authors.jsonl (everyone ever seen posting, including departed
    users); ID lookup falls through to a global user fetch.

    Returns (target, error_message). `target` is a discord.Member when the user is
    still in the guild, otherwise a discord.User.
    """
    authors = authors or {"by_id": {}, "by_name": {}}

    if isinstance(target_raw, int):
        target_id = target_raw
    else:
        m = _COLLECT_MENTION_RE.match(target_raw)
        if m:
            target_id = int(m.group(1))
        elif target_raw.isdigit():
            target_id = int(target_raw)
        else:
            key   = target_raw.lower()
            match = identity["by_name"].get(key) or authors["by_name"].get(key)
            if match is None:
                if authors["by_name"]:
                    hint = (f"{len(authors['by_id'])} user(s) indexed from history — "
                            f"they have not posted in any scanned channel. "
                            f"Pass their user ID or @mention.")
                else:
                    # The index has never been built, so a departed user cannot
                    # resolve by name yet. Point at the command that builds it.
                    hint = ("No history index yet — run `/index` first "
                            "to make users who have left the server resolvable by name.")
                return None, (
                    f"_No member found with name `{target_raw}` "
                    f"({member_count} current members). {hint}_"
                )
            target_id = int(match["user_id"])

    member = guild.get_member(target_id)
    if member is not None:
        return member, None
    try:
        return await guild.fetch_member(target_id), None
    except discord.NotFound:
        pass
    except discord.HTTPException as exc:
        # Rate limit or transient failure — fall through to the global fetch
        # rather than aborting the whole collect run.
        logger.warning("fetch_member(%s) failed on guild %s: %s", target_id, guild.id, exc)

    # Not a current member — resolve globally so their history is still collectable.
    try:
        user = await client.fetch_user(target_id)
    except discord.NotFound:
        return None, f"_No Discord user exists with ID `{target_id}`._"
    except discord.HTTPException as exc:
        return None, f"_Could not look up user ID `{target_id}`: {exc}_"
    logger.info("Collect target %s (%s) is no longer in guild %s — using global user record",
                target_id, user.name, guild.id)
    return user, None


async def _handle_index(cmd: str, message: discord.Message) -> None:
    """Sweep channels and record every user who has posted, storing no messages.

    Separate from /collect because the index must be buildable *before* any
    target has been resolved: the index is what makes a user who has left the
    server resolvable by name, so requiring a resolved target to build it would
    be circular.
    """
    if message.guild is None:
        await message.channel.send("_`/index` only works inside a server._")
        return

    invoker = message.author
    if not (invoker.guild_permissions.manage_messages
            or invoker.guild_permissions.administrator):
        await message.channel.send("_`/index` requires **Manage Messages** permission._")
        return

    _, channel_filter, since, limit, errors = _parse_collect_flags(cmd, message.guild,
                                                                   require_targets=False)
    for err in errors:
        await message.channel.send(err)
    if errors:
        return

    guild_id = str(message.guild.id)
    status   = await message.channel.send("_Indexing server members..._")
    await _scan_members(message.guild, guild_id)
    channels = [channel_filter] if channel_filter else list(message.guild.text_channels)
    _, _, skipped, _, _ = await _scan_user_channels(
        message.guild, guild_id, None, channels, since, limit, set(), status,
        label="index",
    )

    index    = _collect_history.load_authors(guild_id)
    identity = _collect_history.load_identity(guild_id)
    departed = [r for uid, r in index["by_id"].items()
                if uid not in identity["by_id"] and not r.get("is_bot")]
    lines = [
        "**User index built.**",
        f"{len(index['by_id'])} user(s) seen posting"
        f"{f' ({len(skipped)} channel(s) skipped)' if skipped else ''}.",
    ]
    if departed:
        shown = ", ".join(f"`{r.get('username', r['user_id'])}`" for r in departed[:15])
        more  = f" (+{len(departed) - 15} more)" if len(departed) > 15 else ""
        lines.append(f"{len(departed)} no longer in the server: {shown}{more}")
        lines.append("_These are now collectable by name with `/collect`._")
    await status.edit(content="\n".join(lines))


async def _handle_collect(cmd: str, message: discord.Message) -> None:
    """Collect messages for one or more users and save to UserRecognition dataset."""
    if message.guild is None:
        await message.channel.send("_`/collect` only works inside a server._")
        return

    targets, channel_filter, since, limit, errors = _parse_collect_flags(cmd, message.guild)
    for err in errors:
        await message.channel.send(err)
    if errors or not targets:
        return

    guild_id = str(message.guild.id)
    invoker  = message.author
    has_perm = (
        invoker.guild_permissions.manage_messages
        or invoker.guild_permissions.administrator
    )

    # ── Expand --batch into target list from targets.txt ─────────────────────
    if targets == ["__BATCH__"]:
        if not has_perm:
            await message.channel.send("_`--batch` requires **Manage Messages** permission._")
            return
        targets_file = _collect_history._DATA_ROOT / guild_id / "targets.txt"
        if not targets_file.exists():
            await message.channel.send(
                f"_No `targets.txt` found for this guild.\n"
                f"Create `UserRecognition/0-Data/data/{guild_id}/targets.txt` "
                "with one username or user ID per line (lines starting with `#` are comments)._"
            )
            return
        raw_lines = targets_file.read_text(encoding="utf-8").splitlines()
        targets = [ln.strip() for ln in raw_lines if ln.strip() and not ln.startswith("#")]
        if not targets:
            await message.channel.send("_`targets.txt` is empty (or all lines are comments)._")
            return

    # Collecting multiple users always requires Manage Messages.
    multi = len(targets) > 1
    if multi and not has_perm:
        await message.channel.send("_Collecting multiple users requires **Manage Messages** permission._")
        return

    # ── Index members once, then iterate ─────────────────────────────────────
    status = await message.channel.send("_Indexing server members..._")
    member_count = await _scan_members(message.guild, guild_id)
    identity     = _collect_history.load_identity(guild_id)
    authors      = _collect_history.load_authors(guild_id)
    seen_ids     = _collect_history.load_seen_ids(guild_id)
    channels     = [channel_filter] if channel_filter else list(message.guild.text_channels)
    results: list[str] = []

    for i, t_raw in enumerate(targets, 1):
        if multi:
            await status.edit(content=f"_Collecting {i}/{len(targets)}: {t_raw}..._")

        target, err = await _resolve_collect_target(
            t_raw, message.guild, identity, member_count, authors
        )
        if err:
            results.append(f"❌ `{t_raw}` — {err.strip('_')}")
            continue

        # discord.Member only exists for current members; a User means they left.
        departed = not isinstance(target, discord.Member)

        # Single-user: allow self-collection without Manage Messages.
        if not multi and not has_perm and invoker.id != target.id:
            await message.channel.send(
                "_You need **Manage Messages** permission to collect another user's history._"
            )
            return

        if not multi:
            note = " (no longer in this server)" if departed else ""
            await status.edit(
                content=f"_Collecting messages from **{target.display_name}**{note}..._"
            )

        msg_found, msg_new, skipped, first_ts, last_ts = await _scan_user_channels(
            message.guild, guild_id, target.id, channels, since, limit, seen_ids, status,
            label=target.display_name if multi else "",
        )
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _collect_history.save_user({
            "user_id":           str(target.id),
            "guild_id":          guild_id,
            "username":          target.name,
            "display_name":      target.display_name,
            "in_guild":          not departed,
            "message_count":     msg_found,
            "first_message_at":  first_ts or now_iso,
            "last_message_at":   last_ts  or now_iso,
            "last_collected_at": now_iso,
        }, guild_id)
        skip_note = f" ({len(skipped)} skipped)" if skipped else ""
        left_note = " _(left server)_" if departed else ""
        results.append(
            f"✅ **{target.display_name}**{left_note} — {msg_new} new / {msg_found} found{skip_note}"
        )

    header = "**Collection complete:**" if multi else f"**Collected: {targets[0]}**"
    await status.edit(content=header + "\n" + "\n".join(results))


async def _handle_identify(text: str, message: discord.Message) -> None:
    """Rank likely authors of text using the guild's trained user recognition model."""
    if message.guild is None:
        await message.channel.send("_`/identify` only works inside a server._")
        return

    guild_id = str(message.guild.id)
    if not _identify.model_exists(guild_id):
        await message.channel.send(
            "_No trained model found for this server. "
            "Collect message history with `/collect`, then run `train.py --guild " + guild_id + "` to train._"
        )
        return

    try:
        results = _identify.identify(text, guild_id)
    except Exception as exc:
        logger.error("identify error: %s", exc)
        await message.channel.send("_Error running user recognition model — check logs._")
        return

    lines = [f"**Likely authors of:** _{text[:120]}{'…' if len(text) > 120 else ''}_"]
    for rank, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        lines.append(f"`{rank}.` **{r['username']}** {bar} {r['score']*100:.1f}%")
    await message.channel.send("\n".join(lines))


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
            "/history — Show your conversation history\n"
            "/index [--channel #ch] [--since YYYY-MM-DD] [--limit N] — Index everyone who has posted "
            "(run once; needed to reach users who have left)\n"
            "/collect @user [--channel #ch] [--since YYYY-MM-DD] [--limit N] — Save a user's message history\n"
            "/collect \"Display Name\" — Quote names containing spaces\n"
            "/collect --batch — Collect all users listed in data/{guild_id}/targets.txt\n"
            "/identify <text> — Rank likely senders of a message using the trained user recognition model"
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

    if msg.startswith("/index"):
        await _handle_index(msg[len("/index"):].strip(), message)
        return

    if msg.startswith("/collect"):
        await _handle_collect(msg[len("/collect"):].strip(), message)
        return

    if msg.startswith("/identify"):
        text = msg[len("/identify"):].strip()
        if not text:
            await message.channel.send("_Usage: `/identify <text>` — rank likely senders of a message._")
        else:
            await _handle_identify(text, message)
        return

    await message.channel.send("Unrecognized command. Type `/help` for a list of available commands.")


# Guarded so that importing this module does not connect to Discord. Without
# the guard any import — a test, a REPL, a tooling check — starts a second live
# bot that races the real one: both answer every command, and both append to the
# same collection files, where the in-memory seen_ids dedup cannot see the other
# process's writes.
if __name__ == "__main__":
    # Token loaded from DISCORD_BOT_TOKEN in .env or environment
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Add it to your .env file.")

    client.run(bot_token, log_handler=None)
