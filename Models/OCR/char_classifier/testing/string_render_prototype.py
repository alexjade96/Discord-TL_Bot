"""Prototype: string-level rendering as a replacement for TileGrid3x3 tiling.

Motivation (see confused_pairs_latin_run4.json / handoff discussion,
2026-08-16): TileGrid3x3 tiles a single centered glyph into a 3x3 grid and
crops with RandomResizedCrop to fake "glyph in word context". Widening the
crop scale (0.20-0.45 -> 0.40-0.75) reduced but did not remove a collapse
pattern -- it just partially relocated the collapse target (low_i -> low_v),
consistent with the crop landing on ambiguous stroke fragments that don't
correspond to any real neighboring glyph, since the "neighbor" is just a
rotated/mirrored copy of the same character.

This prototype renders a short real string (the target character plus
genuine, different neighbor characters) in one draw call -- the same shape
of pipeline EasyOCR's own training data (MJSynth/SynthText via
TextRecognitionDataGenerator) uses, word-level rather than isolated-glyph.
Per-character bounding boxes are derived from cumulative font advance
widths, then a crop window is taken around the TARGET character only,
sized to roughly the same 1.2-2.3 char-width range TileGrid3x3's widened
crop scale currently produces on a 3x tile. Because the neighbors are real,
different characters, a crop can never land on an "ownerless" fragment --
it either falls inside the target glyph's true ink or shows genuine
neighboring content, matching how Chars74K-style real segmented crops work.

This is a visual/qualitative prototype only -- it does NOT wire into
data.py's dataloader or render_chars.py's dataset build. Run it, inspect
the saved crops, and decide whether to integrate before touching the
production pipeline.

Run from Models/OCR/:
    python -m char_classifier.testing.string_render_prototype
    python -m char_classifier.testing.string_render_prototype --targets cap_W low_v low_i --samples 5
"""
import argparse
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE           = Path(__file__).parent
_FONTS_DIR      = _HERE.parent.parent.parent / 'Datasets' / 'windows-fonts'
_OUT_DIR        = _HERE / 'string_render_samples'
_OUT_SIZE       = 224
_RENDER_SIZE_PT = 64          # font point size for the rendered string
_CANVAS_PAD     = 200         # generous padding so crops never run off-canvas

# label -> literal character, mirroring render_chars.py's naming so results
# are directly comparable to confused_pairs output.
_LABEL_TO_CHAR = {
    **{f'cap_{c}': c for c in string.ascii_uppercase},
    **{f'low_{c}': c for c in string.ascii_lowercase},
    **{f'dig_{c}': c for c in string.digits},
}

# Default targets: the classes actually implicated in run 4's top confused
# pairs (both the low_i side and the new low_v collapse) -- see
# confused_pairs_latin_run4.json.
_DEFAULT_TARGETS = ['low_i', 'low_v', 'cap_W', 'low_w', 'cap_V', 'dig_0', 'low_o', 'cap_O']

_NEIGHBOR_POOL = string.ascii_letters + string.digits


def _list_usable_fonts(n: int, seed: int) -> list[Path]:
    all_fonts = sorted(_FONTS_DIR.glob('*.[tT][tT][fF]')) + sorted(_FONTS_DIR.glob('*.[oO][tT][fF]'))
    rng = random.Random(seed)
    rng.shuffle(all_fonts)
    return all_fonts[:n]


def _char_offsets(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> list[tuple[float, float]]:
    """Cumulative advance-width offsets (start_x, end_x) for each char in text."""
    offsets = []
    x0 = 0.0
    for i in range(len(text)):
        x1 = draw.textlength(text[:i + 1], font=font)
        offsets.append((x0, x1))
        x0 = x1
    return offsets


def render_string_crop(font_path: Path, target_char: str, rng: random.Random,
                       dark: bool) -> Image.Image | None:
    """Render a short string containing target_char among real neighbors,
    then crop a window around target_char only. Returns None if the font
    can't render the target glyph."""
    try:
        font = ImageFont.truetype(str(font_path), _RENDER_SIZE_PT)
    except Exception:
        return None

    n_before = rng.randint(1, 3)
    n_after  = rng.randint(1, 3)
    neighbors_before = ''.join(rng.choice(_NEIGHBOR_POOL) for _ in range(n_before))
    neighbors_after  = ''.join(rng.choice(_NEIGHBOR_POOL) for _ in range(n_after))
    text = neighbors_before + target_char + neighbors_after
    target_idx = n_before

    bg_val = 30  if dark else 255
    fg_val = 255 if dark else 0

    canvas_w = _CANVAS_PAD * 2
    canvas_h = _CANVAS_PAD * 2
    canvas = Image.new('L', (canvas_w, canvas_h), bg_val)
    draw = ImageDraw.Draw(canvas)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None  # font has no usable glyphs for this string

    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    origin_x = _CANVAS_PAD - bbox[0]
    origin_y = _CANVAS_PAD - bbox[1]
    draw.text((origin_x, origin_y), text, font=font, fill=fg_val)

    offsets = _char_offsets(draw, text, font)
    t_start, t_end = offsets[target_idx]
    target_w = max(t_end - t_start, 4.0)
    target_cx = origin_x + (t_start + t_end) / 2.0
    target_cy = origin_y + text_h / 2.0 + bbox[1]

    # A single glyph's own ink bbox can be empty for whitespace-like glyphs;
    # skip if the target character itself didn't actually draw anything --
    # cheap check via cropping just its advance slot before windowing.
    target_slot = canvas.crop((
        int(origin_x + t_start), int(origin_y + bbox[1]),
        int(origin_x + t_end),   int(origin_y + bbox[1] + text_h),
    ))
    if target_slot.getbbox() is None:
        return None

    # Crop window: 1.2-2.3x the target char's own advance width, matching
    # the range TileGrid3x3's widened crop scale (0.40-0.75 on a 3x tile)
    # currently produces -- so results are comparable apples-to-apples.
    win = target_w * rng.uniform(1.2, 2.3)
    half = win / 2.0
    left, top    = target_cx - half, target_cy - half
    right, bottom = target_cx + half, target_cy + half

    crop = canvas.crop((int(left), int(top), int(right), int(bottom)))
    crop = crop.resize((_OUT_SIZE, _OUT_SIZE), Image.LANCZOS)

    rgb_bg = (bg_val, bg_val, bg_val)
    out = Image.new('RGB', (_OUT_SIZE, _OUT_SIZE), rgb_bg)
    out.paste(Image.merge('RGB', [crop, crop, crop]), (0, 0))
    return out


def main():
    p = argparse.ArgumentParser(description='Prototype string-level render + target-glyph crop')
    p.add_argument('--targets', nargs='+', default=_DEFAULT_TARGETS,
                   help='Class labels to render samples for (render_chars.py naming)')
    p.add_argument('--samples', type=int, default=6, help='Samples per target class')
    p.add_argument('--fonts', type=int, default=8, help='Number of random fonts to draw from')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=str(_OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    fonts = _list_usable_fonts(args.fonts, args.seed)
    print(f'[string_render_prototype] {len(fonts)} candidate fonts, '
          f'{len(args.targets)} target classes, {args.samples} samples each')

    saved, skipped = 0, 0
    for label in args.targets:
        ch = _LABEL_TO_CHAR.get(label)
        if ch is None:
            print(f'  [skip] unknown label {label!r}')
            continue
        made = 0
        attempts = 0
        while made < args.samples and attempts < args.samples * 6:
            attempts += 1
            font_path = rng.choice(fonts)
            dark = rng.random() < 0.5
            img = render_string_crop(font_path, ch, rng, dark)
            if img is None:
                skipped += 1
                continue
            fname = f'{label}_{font_path.stem}_{made}.png'
            img.save(out_dir / fname)
            made += 1
            saved += 1
        if made < args.samples:
            print(f'  [warn] {label}: only {made}/{args.samples} rendered '
                  f'(font coverage or empty-glyph skips)')

    print(f'\n[string_render_prototype] saved {saved} sample crops to {out_dir} '
          f'({skipped} render attempts skipped)')


if __name__ == '__main__':
    main()
