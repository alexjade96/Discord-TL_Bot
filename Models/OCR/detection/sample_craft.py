"""
sample_craft.py — Generate a test image and run CRAFT text detection on it.

Produces:
  sample_craft_annotated.png   — original image with detected boxes overlaid
  sample_craft_crops/          — each detected region saved as a numbered crop

Run from OCR/:
    python -m detection.sample_craft
    python -m detection.sample_craft --text-threshold 0.5 --link-threshold 0.3
"""

import sys, argparse, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.craft_detector import detect, detect_and_crop

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Colour palette for box outlines (cycles if more regions than colours)
# ---------------------------------------------------------------------------
_PALETTE = [
    (220,  50,  50),   # red
    ( 50, 160,  50),   # green
    ( 50, 100, 220),   # blue
    (200, 130,  20),   # amber
    (160,  50, 200),   # purple
    ( 30, 180, 180),   # teal
]


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _build_test_image() -> Image.Image:
    """
    Synthesise a Discord-style screenshot with mixed content:
      - English sentence (serif)
      - Korean phrase
      - Japanese phrase
      - Chinese phrase
      - Mixed Latin + digits (username-style)
      - Punctuation-heavy English
    All rendered on a dark #2f3136 background, white text, like a dark-mode
    Discord message pane.
    """
    W, H   = 640, 320
    BG     = (47, 49, 54)     # Discord dark
    FG     = (220, 221, 222)  # Discord text grey
    ACCENT = (114, 137, 218)  # Discord blurple (username colour)

    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    arial      = _load_font('C:/Windows/Fonts/arial.ttf',    18)
    arial_bold = _load_font('C:/Windows/Fonts/arialbd.ttf',  18)
    arial_lg   = _load_font('C:/Windows/Fonts/arial.ttf',    22)
    cjk_font   = _load_font('C:/Windows/Fonts/malgun.ttf',   18)   # Malgun Gothic covers KO/JA/ZH

    lines = [
        # (x, y, text, font, colour)
        ( 20,  20, 'DiscordUser#1234',            arial_bold, ACCENT),
        ( 20,  50, 'Hello! This is an English test sentence.',  arial, FG),
        ( 20,  90, 'discorduser_42',               arial,      ACCENT),
        ( 20, 120, '안녕하세요, 반갑습니다.',               cjk_font,   FG),
        ( 20, 160, 'AnotherUser',                  arial_bold, ACCENT),
        ( 20, 190, 'こんにちは世界！ テスト文字列。',            cjk_font,   FG),
        ( 20, 230, 'user_xyz',                     arial_bold, ACCENT),
        ( 20, 260, '你好，这是一个中文测试。 Hello world 123.',  cjk_font,   FG),
    ]

    for x, y, text, font, colour in lines:
        draw.text((x, y), text, font=font, fill=colour)

    return img


def _draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    out  = image.copy()
    draw = ImageDraw.Draw(out)
    label_font = _load_font('C:/Windows/Fonts/arial.ttf', 10)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        colour = _PALETTE[i % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), str(i + 1),
                  font=label_font, fill=colour)

    return out


def main():
    p = argparse.ArgumentParser(description='CRAFT detection sample')
    p.add_argument('--image',          default=None,
                   help='Path to an existing image (uses synthetic if omitted)')
    p.add_argument('--text-threshold', type=float, default=0.6)
    p.add_argument('--link-threshold', type=float, default=0.3)
    p.add_argument('--low-text',       type=float, default=0.35)
    p.add_argument('--long-size',      type=int,   default=1280)
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Build or load source image
    # -----------------------------------------------------------------------
    if args.image:
        source = Image.open(args.image).convert('RGB')
        print(f'[sample] Source image: {args.image}  ({source.size[0]}×{source.size[1]}px)')
    else:
        source = _build_test_image()
        print(f'[sample] Using synthetic test image ({source.size[0]}×{source.size[1]}px)')

    src_path = HERE / 'sample_craft_source.png'
    source.save(src_path)
    print(f'[sample] Source saved: {src_path}')

    # -----------------------------------------------------------------------
    # Detect
    # -----------------------------------------------------------------------
    print(f'[sample] Running CRAFT  '
          f'(text={args.text_threshold}  link={args.link_threshold}  '
          f'low={args.low_text}) ...')

    boxes = detect(
        source,
        text_threshold = args.text_threshold,
        link_threshold = args.link_threshold,
        low_text       = args.low_text,
        long_size      = args.long_size,
    )
    print(f'[sample] {len(boxes)} region(s) detected')

    if not boxes:
        print('[sample] No regions detected — try lowering --text-threshold')
        return

    # -----------------------------------------------------------------------
    # Annotated output
    # -----------------------------------------------------------------------
    annotated     = _draw_boxes(source, boxes)
    annotated_out = HERE / 'sample_craft_annotated.png'
    annotated.save(annotated_out)
    print(f'[sample] Annotated image: {annotated_out}')

    # -----------------------------------------------------------------------
    # Crops
    # -----------------------------------------------------------------------
    crops_dir = HERE / 'sample_craft_crops'
    crops_dir.mkdir(exist_ok=True)
    crops = detect_and_crop(
        source,
        text_threshold = args.text_threshold,
        link_threshold = args.link_threshold,
        low_text       = args.low_text,
        long_size      = args.long_size,
    )
    for i, crop in enumerate(crops, 1):
        out = crops_dir / f'region_{i:03d}.png'
        crop.save(out)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    print(f'{"#":>4}  {"Box (x1,y1,x2,y2)":30}  {"Size":12}')
    print('-' * 54)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        print(f'{i:>4}  ({x1:4d},{y1:4d})->({x2:4d},{y2:4d})    '
              f'{x2-x1:3d}x{y2-y1:3d}px')
    print()
    print(f'[sample] Crops saved to: {crops_dir}/  ({len(crops)} files)')


if __name__ == '__main__':
    main()
