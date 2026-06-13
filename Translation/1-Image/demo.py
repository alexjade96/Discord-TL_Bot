"""Full pipeline demonstration using TRDG-generated sample images.

Generates synthetic text images in multiple languages, runs each through
OCR → translation → collection, then summarises the resulting dataset.

Usage:
    python demo.py
    python demo.py --no-collect    # skip saving to data/
    python demo.py --save-images   # write generated PNGs to demo_output/
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Force UTF-8 output so CJK characters print correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Pipeline imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "2-Text"))

FONT_DIR = "C:/Windows/Fonts"

# fmt: off
SAMPLES = [
    {
        "label":    "Chinese (Simplified)",
        "strings":  ["早上好", "谢谢你", "你好世界", "今天天气很好"],
        "font":     f"{FONT_DIR}/simsun.ttc",
        "expected": "Chinese",
    },
    {
        "label":    "Japanese",
        "strings":  ["おはようございます", "ありがとう", "東京は美しい", "今日はいい天気です"],
        "font":     f"{FONT_DIR}/msgothic.ttc",
        "expected": "Japanese",
    },
    {
        "label":    "Korean",
        "strings":  ["안녕하세요", "감사합니다", "서울에 오신 것을 환영합니다", "오늘 날씨가 좋네요"],
        "font":     f"{FONT_DIR}/malgun.ttf",
        "expected": "Korean",
    },
    {
        "label":    "Mixed (English + Korean)",
        "strings":  ["hello 안녕 world", "Game over 게임 오버", "Level up 레벨 업"],
        "font":     f"{FONT_DIR}/malgun.ttf",
        "expected": "Korean",
    },
    {
        "label":    "English",
        "strings":  ["Good morning", "Hello world", "How are you today?"],
        "font":     f"{FONT_DIR}/arial.ttf",
        "expected": "English (passthrough)",
    },
]
# fmt: on


def generate_images(strings: list[str], font_path: str, font_size: int = 48) -> list[np.ndarray]:
    """Render text strings to BGR numpy arrays using PIL with the given font.

    Adds a light Gaussian noise background to simulate real-world conditions.
    Uses the modern Pillow 10+ API (getbbox instead of removed getsize).
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    images = []
    for text in strings:
        # Measure text bounds with modern API
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0] + 20
        h = bbox[3] - bbox[1] + 20

        # Gaussian noise background (simulates real image texture)
        noise = np.random.normal(240, 8, (h, w, 3)).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(noise)

        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10 - bbox[1]), text, font=font, fill=(20, 20, 20))

        # Light blur to make it more realistic
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(arr)

    return images


def run_demo(collect: bool = True, save_images: bool = False) -> None:
    from translate_image import translate_image
    from collect import dataset_stats

    output_dir = Path(__file__).parent / "demo_output"
    if save_images:
        output_dir.mkdir(exist_ok=True)

    total_passed = 0
    total_failed = 0

    for group in SAMPLES:
        print(f"\n{'-' * 60}")
        print(f"  {group['label']}  (expected -> {group['expected']})")
        print(f"{'-' * 60}")

        images = generate_images(group["strings"], group["font"], font_size=48)

        for i, (text_str, img) in enumerate(zip(group["strings"], images)):
            if save_images:
                out_path = output_dir / f"{group['label'].replace(' ', '_')}_{i}.png"
                cv2.imwrite(str(out_path), img)

            try:
                # Pass raw numpy array directly — skips URL download
                result = translate_image(img) if collect else _translate_no_collect(img)

                method_tag = f"[{result['method']}]"
                lang = result["source_language"]
                conf = result.get("confidence")
                ocr_conf = result.get("ocr_confidence", 0)
                original = result["original_text"] or "(no text detected)"
                translated = result["translated_text"] or "(none)"

                conf_str = f" {conf*100:.0f}%" if conf is not None else ""
                print(f"\n  Input string : {text_str!r}")
                print(f"  OCR output   : {original!r}  (OCR conf: {ocr_conf*100:.0f}%)")
                print(f"  Translation  : {translated!r}")
                print(f"  Language     : {lang}{conf_str}  {method_tag}")
                total_passed += 1

            except Exception as e:
                print(f"\n  Input string : {text_str!r}")
                print(f"  ERROR        : {e}")
                total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {total_passed} passed / {total_failed} failed")

    if collect:
        stats = dataset_stats()
        print(f"\n  Dataset after demo:")
        print(f"    Total collected : {stats['total']} images")
        print(f"    Breakdown       : {stats['languages']}")
        print(f"    Location        : {stats['images_dir']}")

    print(f"{'=' * 60}\n")


def _translate_no_collect(img: np.ndarray) -> dict:
    """Run OCR + translation without saving to the dataset."""
    from ocr import extract_text_combined
    from translate_text import translate_text

    original_text, ocr_confidence = extract_text_combined(img)
    if not original_text:
        return {
            "original_text": "",
            "translated_text": "",
            "source_language": "unknown",
            "confidence": None,
            "ocr_confidence": 0.0,
            "method": "none",
        }
    result = translate_text(original_text)
    return {
        "original_text": original_text,
        "translated_text": result["translated_text"],
        "source_language": result["source_language"],
        "confidence": result["confidence"],
        "ocr_confidence": ocr_confidence,
        "method": result["method"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline demo with TRDG images.")
    parser.add_argument("--no-collect", action="store_true", help="Skip saving images to data/")
    parser.add_argument("--save-images", action="store_true", help="Write generated PNGs to demo_output/")
    args = parser.parse_args()

    print("\nTL-Bot Full Pipeline Demo")
    print("Generating synthetic images -> OCR -> Translate -> Collect\n")
    print("Note: First run loads EasyOCR models (~30s). Subsequent runs are fast.\n")

    run_demo(collect=not args.no_collect, save_images=args.save_images)
