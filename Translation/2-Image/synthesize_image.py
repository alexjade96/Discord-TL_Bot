"""Image synthesis: erase original text and render translated text in-place.

Given the original image, OCR segments (with bounding boxes from extract_text()),
and the translated text, this module:
  1. Erases each OCR text region by filling with the estimated background colour.
  2. Renders the translated text into the union bounding area, using the largest
     font size that fits with automatic word wrapping.

Bounding boxes from extract_text() are in original image coordinate space
(ocr.py handles the scale conversion internally via _PREPROCESS_SCALE).

Usage (CLI):
    python synthesize_image.py image.png "Translated text here"
    python synthesize_image.py image.png "번역된 텍스트" --lang ko --out out.png
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Per-language ordered font candidate list (Windows system paths).
# First match wins; always falls through to the Latin fallback list.
_FONT_CANDIDATES_BY_LANG: dict[str, list[str]] = {
    "ko":    ["C:/Windows/Fonts/malgun.ttf"],
    "ja":    ["C:/Windows/Fonts/YuGothR.ttc",
               "C:/Windows/Fonts/msgothic.ttc",
               "C:/Windows/Fonts/meiryo.ttc"],
    "zh-cn": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"],
    "zh-tw": ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/mingliu.ttc"],
    "zh":    ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"],
}
_FONT_CANDIDATES_LATIN: list[str] = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]

# ITU-R BT.601 luma coefficients for perceived brightness (standard RGB → Y).
_LUMA_R: float = 0.299
_LUMA_G: float = 0.587
_LUMA_B: float = 0.114
# Luminance midpoint: backgrounds above this are considered light (use dark text).
_LUMA_THRESHOLD: int = 128

# Font size search range and step for _fit_text.
_FONT_MAX_SIZE: int = 72
_FONT_MIN_SIZE: int = 8
_FONT_SIZE_STEP: int = 2   # decrement per iteration; smaller = finer but slower

# Pixel margin sampled outside each text bbox to estimate the background colour.
_BG_SAMPLE_MARGIN: int = 5


def _font_candidates(lang: str) -> list[str]:
    lang_fonts = _FONT_CANDIDATES_BY_LANG.get(lang.lower(), [])
    return lang_fonts + _FONT_CANDIDATES_LATIN


def _load_font(size: int, lang: str = "en") -> ImageFont.FreeTypeFont:
    """Load the best available TrueType font for lang at the given size."""
    for path in _font_candidates(lang):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _bbox_to_rect(bbox: list) -> tuple[int, int, int, int]:
    """Convert EasyOCR [[x,y],...] corners → (x1, y1, x2, y2) integer rect."""
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _sample_background(
    img_rgb: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    margin: int = _BG_SAMPLE_MARGIN,
) -> tuple[int, int, int]:
    """Estimate background colour from the ring of pixels just outside the bbox."""
    h, w = img_rgb.shape[:2]
    ox1 = max(0, x1 - margin)
    oy1 = max(0, y1 - margin)
    ox2 = min(w, x2 + margin)
    oy2 = min(h, y2 + margin)
    outer = img_rgb[oy1:oy2, ox1:ox2]

    # Build a mask for the border ring (outer minus inner).
    mask = np.ones(outer.shape[:2], dtype=bool)
    iy1 = max(0, y1 - oy1)
    iy2 = max(0, y2 - oy1)
    ix1 = max(0, x1 - ox1)
    ix2 = max(0, x2 - ox1)
    mask[iy1:iy2, ix1:ix2] = False

    border = outer[mask]
    if border.size == 0:
        return (255, 255, 255)
    median = np.median(border, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


def _contrasting_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white, whichever contrasts more with bg."""
    luminance = _LUMA_R * bg[0] + _LUMA_G * bg[1] + _LUMA_B * bg[2]
    return (0, 0, 0) if luminance > _LUMA_THRESHOLD else (255, 255, 255)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    lang: str = "en",
    max_size: int = _FONT_MAX_SIZE,
    min_size: int = _FONT_MIN_SIZE,
) -> tuple[ImageFont.FreeTypeFont, str]:
    """Binary-search for the largest font size where wrapped text fits in the box."""
    for size in range(max_size, min_size - 1, -_FONT_SIZE_STEP):
        font = _load_font(size, lang)
        try:
            avg_w = font.getlength("A")
        except AttributeError:
            avg_w = size * 0.6
        avg_w = max(avg_w, 1.0)
        chars_per_line = max(1, int(box_w / avg_w))
        wrapped = textwrap.fill(text, width=chars_per_line)
        try:
            tb = draw.textbbox((0, 0), wrapped, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(wrapped, font=font)  # type: ignore[attr-defined]
        if tw <= box_w and th <= box_h:
            return font, wrapped

    # Smallest size — return even if it overflows (better than nothing).
    font = _load_font(min_size, lang)
    avg_w = max(getattr(font, "getlength", lambda _: min_size * 0.6)("A"), 1.0)
    return font, textwrap.fill(text, width=max(1, int(box_w / avg_w)))


def _load_source(source: bytes | str | Path) -> Image.Image:
    """Load the source image as a PIL RGB Image."""
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    img_cv = cv2.imread(str(source))
    if img_cv is None:
        raise FileNotFoundError(f"Could not load image: {source}")
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def synthesize_image(
    source: bytes | str | Path,
    ocr_segments: list[dict],
    translated_text: str,
    tgt_lang: str = "en",
) -> bytes:
    """Replace original OCR text regions with the translated text.

    Erases each detected text region (filling with the estimated background
    colour), then renders the full translated text into the bounding area that
    covers all original text regions, choosing the largest font that fits.

    Args:
        source:          Original image as bytes, file path, or URL.
        ocr_segments:    Segment list from extract_text() — bboxes are in
                         original image coordinate space.
        translated_text: The translated string to render.
        tgt_lang:        Target language code (used to select a matching font).

    Returns:
        PNG image bytes with original text replaced by the translation.

    Raises:
        ValueError: If ocr_segments is empty or translated_text is blank.
    """
    if not ocr_segments:
        raise ValueError("No OCR segments provided — nothing to replace.")
    if not translated_text or not translated_text.strip():
        raise ValueError("Cannot synthesize image with empty translated text.")

    img = _load_source(source)
    img_rgb = np.array(img)          # snapshot for background sampling
    draw = ImageDraw.Draw(img)

    # extract_text() already returns bboxes in original image space.
    rects = [_bbox_to_rect(seg["bbox"]) for seg in ocr_segments]

    # Erase each text region, collecting background colour estimates.
    bg_colors: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2 in rects:
        bg = _sample_background(img_rgb, x1, y1, x2, y2)
        bg_colors.append(bg)
        draw.rectangle([x1, y1, x2, y2], fill=bg)

    # Union bounding box of all text regions.
    ux1 = min(r[0] for r in rects)
    uy1 = min(r[1] for r in rects)
    ux2 = max(r[2] for r in rects)
    uy2 = max(r[3] for r in rects)
    box_w = max(ux2 - ux1, 1)
    box_h = max(uy2 - uy1, 1)

    # Dominant background across all regions → text colour.
    median_bg: tuple[int, int, int] = tuple(
        int(np.median([c[i] for c in bg_colors])) for i in range(3)
    )  # type: ignore[assignment]
    text_color = _contrasting_color(median_bg)

    # Fit and render the translated text.
    font, wrapped = _fit_text(draw, translated_text, box_w, box_h, lang=tgt_lang)
    draw.text((ux1, uy1), wrapped, fill=text_color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    import argparse
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    sys.path.insert(0, str(Path(__file__).parent))
    from ocr import extract_text

    parser = argparse.ArgumentParser(
        description="Replace image text with a translation."
    )
    parser.add_argument("source", help="Image file path")
    parser.add_argument("text", help="Translated text to render")
    parser.add_argument("--lang", default="en", help="Target language code (default: en)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: synthesized.png)")
    args = parser.parse_args()

    print("Running OCR to detect text regions …")
    segments = extract_text(args.source)
    if not segments:
        print("No text detected in the image.")
        sys.exit(1)
    print(f"Found {len(segments)} segment(s)")

    result = synthesize_image(args.source, segments, args.text, tgt_lang=args.lang)
    out_path = Path(args.out) if args.out else Path("synthesized.png")
    out_path.write_bytes(result)
    print(f"Wrote {len(result):,} bytes → {out_path}")
