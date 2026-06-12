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
from utils import get_language_name  # noqa: E402

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
    backupCount=5,  # Rotate through 5 files
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
                "Hello! I am TL-Bot.  Please send me an image with text or include text when @ting me and I will do my best to translate it for you!"
            )
        else:
            if msg.startswith("/help"):
                await message.channel.send(
                    "Available commands:\n/help - Show this help message\n/hello - Greet the bot\n/test - Test the bot\n/translate - Translate text or image"
                )
            elif msg.startswith("/translate") and message.attachments:
                for attachment in message.attachments:
                    if 'image' in attachment.content_type:
                        image_url = attachment.url
                        await message.channel.send(f"Received image: {image_url}")
                        # Here you would add the OCR and translation logic
                    elif 'audio' in attachment.content_type:
                        audio_url = attachment.url
                        await message.channel.send(f"Received audio file: {audio_url}")
                        # Here you would add the audio transcription and translation logic
                    elif 'video' in attachment.content_type:
                        video_url = attachment.url
                        await message.channel.send(f"Received video file: {video_url}")
                        # Here you would add the video transcription and translation logic
                    elif 'text' in attachment.content_type:
                        text_url = attachment.url
                        await message.channel.send(f"Received text file: {text_url}")
                        # Here you would add the text file reading and translation logic
                    else:
                        unknown_url = attachment.url
                        await message.channel.send(
                            f"Unsupported file type: {unknown_url}."
                        )
            elif msg.startswith("/translate") and message.content.strip() != "":
                text_to_translate = msg.replace("/translate", "").strip()
                if not text_to_translate:
                    await message.channel.send("Please provide text after `/translate`.")
                else:
                    status = await message.channel.send("Translating...")
                    try:
                        result = await asyncio.to_thread(translate_text, text_to_translate)
                        lang_name = get_language_name(result["source_language"])
                        translated = result["translated_text"]
                        confidence = result.get("confidence")
                        conf_str = f" ({confidence * 100:.0f}%)" if confidence is not None else ""
                        await status.edit(content=f"**[{lang_name} → English]{conf_str}** {translated}")
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
