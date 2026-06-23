"""
compare.py — Standard OCR vs per-script char_classifier comparison.

For each CRAFT-detected region:
  1. Standard OCR path  : manga-ocr / EasyOCR (via ocr_pipeline.recognize_crop)
  2. Classifier path    : column-project the region crop → per-char crops → char_classifier
  3. Report             : console table + PNG grid (each char crop with both predictions)

Multi-script classifier loading
  Looks for  checkpoints/<script>/best.pt  for each recognised script.
  Falls back to  checkpoints/best.pt  (legacy single-script location).
  Per-script models are optional; missing ones show '?' in the clf column.

Run from OCR/:
    python -m compare --image screenshot.png
    python -m compare --image screenshot.png --save-grid out.png
    python -m compare --image screenshot.png --top-k 3 --no-grid
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

import ocr_pipeline
from char_classifier.segment import split_into_chars

_HERE      = Path(__file__).parent
_CKPT_ROOT = _HERE / 'checkpoints'
_LEGACY_CKPT = _CKPT_ROOT / 'best.pt'

# ---------------------------------------------------------------------------
# Per-script classifier cache
# ---------------------------------------------------------------------------

_clf_cache: dict[str, tuple] = {}   # script → (model, names, device) or (None, None, None)


def _get_clf(script: str) -> tuple:
    """Return (model, class_names, device) for a script, or (None, None, None) if unavailable."""
    if script in _clf_cache:
        return _clf_cache[script]

    import torch
    from char_classifier.model_builder import create_model
    from char_classifier.utils import get_device, load_checkpoint

    # Prefer script-scoped checkpoint; fall back to legacy best.pt
    candidates = [_CKPT_ROOT / script / 'best.pt', _LEGACY_CKPT]
    for ckpt in candidates:
        names_file = ckpt.parent / 'class_names.json'
        if not (ckpt.exists() and names_file.exists()):
            continue

        cfg_file = ckpt.parent / 'config.json'
        backbone = 'dinov2_vits14'
        if cfg_file.exists():
            backbone = json.load(open(cfg_file)).get('backbone', backbone)

        names  = json.load(open(names_file))
        device = get_device()
        model  = create_model(backbone, num_classes=len(names), freeze_base=False)
        model  = model.to(device)
        load_checkpoint(str(ckpt), model, device=device)
        model.eval()
        _clf_cache[script] = (model, names, device)
        src = 'script' if ckpt.parent.name == script else 'legacy'
        print(f'[compare] Loaded {script} classifier ({src}, {backbone}): {len(names)} classes')
        return _clf_cache[script]

    _clf_cache[script] = (None, None, None)
    return _clf_cache[script]


_eval_tf = None


def _get_eval_tf():
    global _eval_tf
    if _eval_tf is None:
        from torchvision import transforms
        _eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _eval_tf


def _classify_crop(crop: Image.Image, script: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Run classifier on a single char crop. Returns [(label, conf), ...]."""
    import torch

    model, names, device = _get_clf(script)
    if model is None:
        return [('?', 0.0)]

    tf     = _get_eval_tf()
    tensor = tf(crop.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    k = min(top_k, len(names))
    values, indices = torch.topk(probs, k)
    return [(names[i.item()], round(v.item(), 4)) for i, v in zip(indices, values)]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CharComparison:
    crop:      Image.Image
    clf_label: str           # top-1 label from classifier, or '?'
    clf_conf:  float         # 0.0 if no model
    clf_top3:  list          # [(label, conf), ...]
    std_char:  str           # aligned char from std OCR string, or '?'
    agrees:    bool          # clf_label == std_char (only valid when neither is '?')


@dataclass
class RegionComparison:
    region_idx:  int
    bbox:        tuple
    std_text:    str         # full string from standard OCR
    clf_text:    str         # assembled from classifier top-1 predictions
    script:      str
    method:      str
    char_count:  int         # number of segmented char crops
    agree_count: int         # chars where clf matches std
    chars:       list        # list[CharComparison]


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def compare_image(
    source: Image.Image,
    top_k: int = 3,
    **recognize_kwargs,
) -> list[RegionComparison]:
    """
    Run standard OCR + per-script char_classifier on every detected region.

    Parameters
    ----------
    source          : PIL Image
    top_k           : how many classifier predictions to store per char
    recognize_kwargs: forwarded to ocr_pipeline.recognize()

    Returns
    -------
    List of RegionComparison, one per detected text region.
    """
    results = ocr_pipeline.recognize(source, **recognize_kwargs)
    regions: list[RegionComparison] = []

    for i, region in enumerate(results):
        x1, y1, x2, y2 = region.bbox
        crop       = source.crop((x1, y1, x2, y2))
        char_crops = split_into_chars(crop)

        std_str = region.text.replace(' ', '')  # strip spaces for 1-to-1 char alignment
        chars: list[CharComparison] = []
        for j, cc in enumerate(char_crops):
            preds     = _classify_crop(cc, region.script, top_k=top_k)
            clf_label = preds[0][0] if preds else '?'
            clf_conf  = preds[0][1] if preds else 0.0
            std_char  = std_str[j] if j < len(std_str) else '?'
            agrees    = clf_label != '?' and std_char != '?' and clf_label == std_char
            chars.append(CharComparison(
                crop      = cc,
                clf_label = clf_label,
                clf_conf  = clf_conf,
                clf_top3  = preds,
                std_char  = std_char,
                agrees    = agrees,
            ))

        clf_text    = ''.join(c.clf_label for c in chars if c.clf_label != '?')
        agree_count = sum(1 for c in chars if c.agrees)
        regions.append(RegionComparison(
            region_idx  = i + 1,
            bbox        = region.bbox,
            std_text    = region.text,
            clf_text    = clf_text,
            script      = region.script,
            method      = region.method,
            char_count  = len(chars),
            agree_count = agree_count,
            chars       = chars,
        ))

    return regions


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(regions: list[RegionComparison]) -> None:
    total_chars  = sum(r.char_count  for r in regions)
    total_agrees = sum(r.agree_count for r in regions)

    for r in regions:
        pct = f'{r.agree_count / r.char_count:.0%}' if r.char_count else 'N/A'
        print(f'\nRegion {r.region_idx}  bbox={r.bbox}  script={r.script}  '
              f'method={r.method}')
        print(f'  Standard OCR : "{r.std_text}"')
        print(f'  Classifier   : "{r.clf_text}"')
        print(f'  Agreement    : {r.agree_count}/{r.char_count} chars ({pct})')
        print(f'  {"#":>3}  {"Std":8}  {"Clf":8}  {"Conf":6}  {"Top-3 predictions"}')
        print(f'  {"-" * 65}')
        for j, c in enumerate(r.chars, 1):
            top3_str = '  '.join(f'{lbl}:{cf:.2f}' for lbl, cf in c.clf_top3[:3])
            if c.agrees:
                mark = '✓'
            elif c.std_char == '?' or c.clf_label == '?':
                mark = '?'
            else:
                mark = '✗'
            print(f'  {j:>3}  {c.std_char:8}  {c.clf_label:8}  {c.clf_conf:.4f}  '
                  f'{top3_str}  {mark}')

    if total_chars:
        print(f'\nOverall: {total_agrees}/{total_chars} chars agreed '
              f'({total_agrees / total_chars:.1%})')
    else:
        print('\nNo chars segmented.')


# ---------------------------------------------------------------------------
# PNG grid rendering
# ---------------------------------------------------------------------------

_CELL_W   = 64
_CELL_PAD = 4
_CROP_H   = 52
_LABEL_H  = 44
_CELL_H   = _CROP_H + _LABEL_H + _CELL_PAD * 2
_MAX_COL  = 20         # wrap after this many chars per row
_HEADER_H = 24

_COL_AGREE    = (60,  180,  60)
_COL_DISAGREE = (220,  60,  60)
_COL_UNKNOWN  = (160, 160, 160)
_COL_BG       = (240, 240, 240)
_COL_HEADER   = (40,  60, 100)


def _try_font(paths: list[str], size: int) -> ImageFont.ImageFont:
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_grid(regions: list[RegionComparison]) -> Image.Image:
    font_sm  = _try_font(['C:/Windows/Fonts/malgun.ttf',
                           'C:/Windows/Fonts/msgothic.ttc',
                           'C:/Windows/Fonts/arial.ttf'], 10)
    font_hdr = _try_font(['C:/Windows/Fonts/malgunbd.ttf',
                           'C:/Windows/Fonts/arialbd.ttf',
                           'C:/Windows/Fonts/arial.ttf'], 11)

    rows_per_region = [
        max(1, (len(r.chars) + _MAX_COL - 1) // _MAX_COL)
        for r in regions
    ]

    img_w = _CELL_W * _MAX_COL + _CELL_PAD * 2
    img_h = sum(
        _HEADER_H + rows * _CELL_H + _CELL_PAD
        for rows in rows_per_region
    ) + _CELL_PAD

    canvas = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    y = _CELL_PAD
    for r, n_rows in zip(regions, rows_per_region):
        pct = f'{r.agree_count}/{r.char_count}'
        hdr = (f'Region {r.region_idx}  [{r.script}/{r.method}]  '
               f'"{r.std_text[:35]}"  ({pct} agree)')
        draw.rectangle([_CELL_PAD, y, img_w - _CELL_PAD, y + _HEADER_H - 2],
                       fill=_COL_HEADER)
        draw.text((_CELL_PAD + 4, y + 4), hdr, font=font_hdr, fill=(255, 255, 255))
        y += _HEADER_H

        for row_i in range(n_rows):
            row_chars = r.chars[row_i * _MAX_COL: (row_i + 1) * _MAX_COL]
            for col_i, c in enumerate(row_chars):
                cx = _CELL_PAD + col_i * _CELL_W
                cy = y

                if c.clf_label == '?' or c.std_char == '?':
                    border = _COL_UNKNOWN
                elif c.agrees:
                    border = _COL_AGREE
                else:
                    border = _COL_DISAGREE

                draw.rectangle([cx, cy, cx + _CELL_W - 1, cy + _CELL_H - 1],
                               fill=_COL_BG, outline=border, width=2)

                thumb_w = _CELL_W - 2 * _CELL_PAD
                thumb_h = _CROP_H - 2 * _CELL_PAD
                thumb   = c.crop.convert('RGB').resize((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(thumb, (cx + _CELL_PAD, cy + _CELL_PAD))

                label_y = cy + _CROP_H + 2
                draw.text((cx + 3, label_y),
                          f'std: {c.std_char}', font=font_sm, fill=(50, 50, 50))
                clf_col = (border if c.clf_label != '?' else _COL_UNKNOWN)
                draw.text((cx + 3, label_y + 14),
                          f'clf: {c.clf_label}', font=font_sm, fill=clf_col)
                draw.text((cx + 3, label_y + 28),
                          f'{c.clf_conf:.0%}', font=font_sm, fill=(100, 100, 100))

            y += _CELL_H
        y += _CELL_PAD

    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    p = argparse.ArgumentParser(
        description='Standard OCR vs char_classifier comparison harness')
    p.add_argument('--image',          required=True,
                   help='Input image path')
    p.add_argument('--top-k',          type=int, default=3,
                   help='Top-k classifier predictions shown per character')
    p.add_argument('--save-grid',      default=None,
                   help='Output PNG path (default: <image>_compare.png)')
    p.add_argument('--no-grid',        action='store_true',
                   help='Skip PNG grid generation')
    p.add_argument('--text-threshold', type=float, default=0.7)
    p.add_argument('--link-threshold', type=float, default=0.4)
    p.add_argument('--low-text',       type=float, default=0.4)
    args = p.parse_args()

    src  = Path(args.image)
    img  = Image.open(src).convert('RGB')
    print(f'[compare] Image: {src}  ({img.size[0]}×{img.size[1]})')

    regions = compare_image(
        img,
        top_k          = args.top_k,
        text_threshold = args.text_threshold,
        link_threshold = args.link_threshold,
        low_text       = args.low_text,
    )

    if not regions:
        print('[compare] No text regions detected.')
        return

    print_report(regions)

    if not args.no_grid:
        grid_path = args.save_grid or str(src.with_name(src.stem + '_compare.png'))
        grid = _render_grid(regions)
        grid.save(grid_path)
        print(f'\n[compare] Grid saved: {grid_path}')


if __name__ == '__main__':
    main()
