"""Text translation via HuggingFace Inference API.

Usage (CLI):
    python translate.py "Меня зовут Вольфганг"
    python translate.py --src ru "Меня зовут Вольфганг"

Environment:
    HF_TOKEN — HuggingFace Inference API token (required)
"""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

from detect import detect_language, is_english

# Multilingual → English translation model supported by hf-inference free tier
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-mul-en"


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="hf-inference", api_key=token)


def translate_to_english(text: str, client: InferenceClient | None = None) -> str:
    """Translate text to English using Helsinki-NLP/opus-mt-mul-en."""
    c = client or _client()
    result = c.translation(text, model=TRANSLATE_MODEL)
    return result.translation_text


def translate_text(text: str, src_lang: str | None = None) -> dict:
    """Detect language and translate text to English.

    Args:
        text:     Input text to translate.
        src_lang: Optional langdetect-style language code (e.g. 'ru', 'zh-cn').
                  Detected automatically when omitted.

    Returns:
        dict with keys:
            translated_text  — English output
            source_language  — detected or provided langdetect code
            method           — 'passthrough' | 'opus-mt' | 'none'
    """
    if not text.strip():
        return {"translated_text": text, "source_language": "unknown", "method": "none"}

    detected = src_lang or detect_language(text)

    if detected == "en" or (not src_lang and is_english(text)):
        return {"translated_text": text, "source_language": "en", "method": "passthrough"}

    translated = translate_to_english(text, _client())
    return {"translated_text": translated, "source_language": detected, "method": "opus-mt"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text to English.")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--src", default=None, help="Source language code (e.g. ru, zh-cn)")
    args = parser.parse_args()

    result = translate_text(args.text, src_lang=args.src)
    print(f"[{result['source_language']} → en via {result['method']}]")
    print(result["translated_text"])
