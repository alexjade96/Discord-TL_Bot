# TL-Bot.py
# Discord bot for handling image/text translations

import asyncio
import os
import sys
import datetime
from pathlib import Path

import discord
import logging
import logging.handlers

# Load .env file if python-dotenv is installed
from dotenv import load_dotenv
load_dotenv()

# Add text translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "2-Text"))
from translate_text import translate_text
from utils import get_language_name, parse_language_hint  # noqa: E402
from detect import analyze_segments  # noqa: E402

# Add image translation package to path
sys.path.insert(0, str(Path(__file__).parent / "Translation" / "1-Image"))
from translate_image import translate_image  # noqa: E402

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


# Bot Functions
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = " ".join(message.content.split()[1:])
    print(f"Received message: {msg}")

    if client.user.mentioned_in(message) and message.content.startswith(client.user.mention):
        if message.author == client.user:
            return

        if not len(msg):
            await message.channel.send(
                "Hello! I am TL-Bot. Please send me an image with text or include text when @-ing me and I will do my best to translate it for you!"
            )
        else:
            if msg.startswith("/help"):
                await message.channel.send(
                    "Available commands:\n"
                    "/help — Show this help message\n"
                    "/translate — Translate text or image\n"
                    "  --from <language>  Hint the source language (also shows auto-detect result)\n"
                    "  --to <language>    Set the output language (default: English)\n"
                    "  --analyze          Show detected language segments alongside the translation"
                )

            elif msg.startswith("/translate") and message.attachments:
                cmd = msg[len("/translate"):].strip()
                _, from_lang, to_lang, analyze, errors = _parse_translate_flags(cmd)

                for err in errors:
                    await message.channel.send(err)

                for attachment in message.attachments:
                    if "image" in attachment.content_type:
                        status = await message.channel.send("Translating image...")
                        try:
                            result = await asyncio.to_thread(
                                translate_image, attachment.url, None, from_lang, to_lang
                            )
                            auto = result["auto"]
                            hint = result["hint"]

                            if auto["method"] == "none":
                                await status.edit(content="No text detected in the image.")
                            else:
                                parts = [_fmt_result(auto, "Auto", to_lang, ocr=True)]
                                if hint and hint["method"] != "none":
                                    hint_label = f"Hint ({get_language_name(from_lang)})"
                                    parts.append(_fmt_result(hint, hint_label, to_lang, ocr=True))
                                elif hint:
                                    parts.append(f"**Hint ({get_language_name(from_lang)})** No text detected with hinted reader.")

                                if analyze and auto["original_text"]:
                                    analysis = await asyncio.to_thread(_fmt_analyze, auto["original_text"])
                                    if analysis:
                                        parts.append(analysis)

                                await status.edit(content="\n\n".join(parts))
                        except Exception as e:
                            logger.exception(f"Image translation error: {e}")
                            await status.edit(content="Image translation failed. Please try again later.")

                    elif "audio" in attachment.content_type:
                        await message.channel.send(f"Received audio file: {attachment.url}")
                        # Here you would add the audio transcription and translation logic
                    elif "video" in attachment.content_type:
                        await message.channel.send(f"Received video file: {attachment.url}")
                        # Here you would add the video transcription and translation logic
                    elif "text" in attachment.content_type:
                        await message.channel.send(f"Received text file: {attachment.url}")
                        # Here you would add the text file reading and translation logic
                    else:
                        await message.channel.send(f"Unsupported file type: {attachment.url}.")

            elif msg.startswith("/translate"):
                cmd = msg[len("/translate"):].strip()
                text_to_translate, from_lang, to_lang, analyze, errors = _parse_translate_flags(cmd)

                for err in errors:
                    await message.channel.send(err)

                if not text_to_translate:
                    await message.channel.send("Please provide text after `/translate`.")
                else:
                    status = await message.channel.send("Translating...")
                    try:
                        # Auto-detect pass (always run)
                        auto_result = await asyncio.to_thread(
                            translate_text, text_to_translate, None, to_lang
                        )
                        parts = [_fmt_result(auto_result, "Auto", to_lang)]

                        # Hinted pass (only when --from is given)
                        if from_lang:
                            hint_result = await asyncio.to_thread(
                                translate_text, text_to_translate, from_lang, to_lang
                            )
                            hint_label = f"Hint ({get_language_name(from_lang)})"
                            parts.append(_fmt_result(hint_result, hint_label, to_lang))

                        if analyze:
                            analysis = await asyncio.to_thread(_fmt_analyze, text_to_translate)
                            if analysis:
                                parts.append(analysis)

                        await status.edit(content="\n\n".join(parts))
                    except Exception as e:
                        logger.exception(f"Translation error: {e}")
                        await status.edit(content="Translation failed. Please try again later.")

            else:
                await message.channel.send(
                    "Translate function not yet implemented, please stay tuned!"
                )


# Token loaded from DISCORD_BOT_TOKEN in .env or environment
bot_token = os.getenv("DISCORD_BOT_TOKEN")
if not bot_token:
    raise RuntimeError("DISCORD_BOT_TOKEN is not set. Add it to your .env file.")

client.run(bot_token, log_handler=None)
