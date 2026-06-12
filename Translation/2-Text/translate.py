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
from utils import get_mbart_code

MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MBART = "en_XX"


def _client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(provider="hf-inference", api_key=token)


def translate_with_mbart(text: str, src_mbart_code: str, client: InferenceClient | None = None) -> str:
    """Translate text to English using mBART-50 (supports 50 languages)."""
    c = client or _client()
    result = c.translation(
        text,
        model=MBART_MODEL,
        src_lang=src_mbart_code,
        tgt_lang=TARGET_MBART,
    )
    return result.translation_text


def translate_with_qwen(text: str, client: InferenceClient | None = None) -> str:
    """Translate text to English via Qwen instruction prompting.

    Used as fallback when the source language is not in mBART-50's supported set.
    """
    c = client or _client()
    prompt = (
        "Translate the following text to English. "
        "Reply with only the translation, no explanation:\n\n"
        f"{text}"
    )
    return c.text_generation(prompt, model=QWEN_MODEL, max_new_tokens=512).strip()


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
            method           — 'passthrough' | 'mbart' | 'qwen' | 'none'
    """
    if not text.strip():
        return {"translated_text": text, "source_language": "unknown", "method": "none"}

    detected = src_lang or detect_language(text)

    if detected == "en" or (not src_lang and is_english(text)):
        return {"translated_text": text, "source_language": "en", "method": "passthrough"}

    mbart_code = get_mbart_code(detected)
    c = _client()

    if mbart_code:
        translated = translate_with_mbart(text, mbart_code, c)
        return {"translated_text": translated, "source_language": detected, "method": "mbart"}

    translated = translate_with_qwen(text, c)
    return {"translated_text": translated, "source_language": detected, "method": "qwen"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text to English.")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--src", default=None, help="Source language code (e.g. ru, zh-cn)")
    args = parser.parse_args()

    result = translate_text(args.text, src_lang=args.src)
    print(f"[{result['source_language']} → en via {result['method']}]")
    print(result["translated_text"])
