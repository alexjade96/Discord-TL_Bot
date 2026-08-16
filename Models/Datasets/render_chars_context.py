"""render_chars_context.py -- generate char-dataset-ctx/ using string-level
rendering + target-glyph cropping, instead of render_chars.py's isolated
centered-glyph tiles.

Background: char_classifier's TileGrid3x3 augmentation tiles a single
centered glyph into a 3x3 grid and crops it to fake "glyph in word
context". A confused-pairs diagnostic on the Latin classifier (2026-08-16)
showed this produces a collapse pattern (ambiguous stroke-fragment crops ->
a generic catch-all class) that widening the crop scale only partially
fixed -- it relocated the collapse target (low_i -> low_v) rather than
removing it, because the "neighbor" content in a tiled crop is just a
rotated/mirrored copy of the same glyph, not a real different character.

This script renders a short string (the target character plus genuine,
different neighbor characters drawn from the same script's charset) in one
font pass -- the same shape of pipeline EasyOCR's own training data
(MJSynth/SynthText via TextRecognitionDataGenerator) uses -- then crops a
window around the target character only, using each character's true
bounding box (derived from cumulative font advance widths). Any content at
a crop's edges therefore belongs to a real neighboring glyph, never an
ownerless fragment.

Reuses render_chars.py's charset builders and font-scanning helpers;
writes to a separate dataset root (char-dataset-ctx/) so the existing
char-dataset/ (what runs 1-4 were trained/compared against) is untouched.

Run from Models/Datasets/:
    python render_chars_context.py --scripts latin
    python render_chars_context.py --scripts latin --variants-per-slot 6
"""
import argparse
import random
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from render_chars import (
    CHARSETS, CHARSET_SCRIPTS, SCRIPT_NAMES,
    _build_kana, _build_hangul, _build_cjk,
    copy_system_fonts, collect_extra_fonts, extract_cmap,
)

_HERE            = Path(__file__).parent

TILE_SIZE       = 128   # on-disk tile size -- matches render_chars.py's convention;
                        # data.py's transforms resize to the model's 224 input at train time
RENDER_SIZE_PT  = 64    # single fixed font point size -- deliberately NOT mixing
                        # 32/96pt like render_chars.py, so this run isolates the
                        # tiling-mechanism change from the still-open
                        # resolution-normalization hypothesis (see handoff)
CANVAS_PAD       = 200
MIN_NEIGHBORS    = 1
MAX_NEIGHBORS    = 3
MIN_CROP_SCALE   = 1.2  # x target glyph's own advance width
MAX_CROP_SCALE   = 2.3  # matches TileGrid3x3's widened crop-scale range, for comparability


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _char_offsets(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    offsets = []
    x0 = 0.0
    for i in range(len(text)):
        x1 = draw.textlength(text[:i + 1], font=font)
        offsets.append((x0, x1))
        x0 = x1
    return offsets


def render_context_tile(font: ImageFont.FreeTypeFont, cmap_set: set, target_ch: str,
                        neighbor_chars: list, rng: random.Random, dark: bool,
                        tile_size: int = TILE_SIZE):
    """Render target_ch inside a short real string and crop around it.
    Returns a tile_size x tile_size RGB image, or None if unrenderable."""
    pool = [c for c in neighbor_chars if ord(c) in cmap_set] or [target_ch]

    n_before = rng.randint(MIN_NEIGHBORS, MAX_NEIGHBORS)
    n_after  = rng.randint(MIN_NEIGHBORS, MAX_NEIGHBORS)
    text = (
        ''.join(rng.choice(pool) for _ in range(n_before))
        + target_ch
        + ''.join(rng.choice(pool) for _ in range(n_after))
    )
    target_idx = n_before

    bg_val = 30  if dark else 255
    fg_val = 255 if dark else 0

    canvas = Image.new('L', (CANVAS_PAD * 2, CANVAS_PAD * 2), bg_val)
    draw = ImageDraw.Draw(canvas)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    text_h    = bbox[3] - bbox[1]
    origin_x  = CANVAS_PAD - bbox[0]
    origin_y  = CANVAS_PAD - bbox[1]
    draw.text((origin_x, origin_y), text, font=font, fill=fg_val)

    offsets = _char_offsets(draw, text, font)
    t_start, t_end = offsets[target_idx]
    target_w  = max(t_end - t_start, 4.0)
    target_cx = origin_x + (t_start + t_end) / 2.0
    target_cy = origin_y + text_h / 2.0 + bbox[1]

    target_slot = canvas.crop((
        int(origin_x + t_start), int(origin_y + bbox[1]),
        int(origin_x + t_end),   int(origin_y + bbox[1] + text_h),
    ))
    if target_slot.getbbox() is None:
        return None  # target glyph didn't actually draw anything (missing in font)

    win = target_w * rng.uniform(MIN_CROP_SCALE, MAX_CROP_SCALE)
    half = win / 2.0
    crop = canvas.crop((
        int(target_cx - half), int(target_cy - half),
        int(target_cx + half), int(target_cy + half),
    ))
    crop = crop.resize((tile_size, tile_size), Image.LANCZOS)

    tile_rgb = Image.new('RGB', (tile_size, tile_size), (bg_val, bg_val, bg_val))
    tile_rgb.paste(Image.merge('RGB', [crop, crop, crop]), (0, 0))
    return tile_rgb


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------

def render_charset_context(font_meta: list, charset: list, update: bool, script: str,
                           variants_per_slot: int, seed: int, out_root: str, tile_size: int):
    """Render all (label, char) pairs in charset to <out_root>/<script>/."""
    script_dir = Path(out_root) / script
    script_dir.mkdir(parents=True, exist_ok=True)

    neighbor_chars = [ch for _, ch in charset]
    font_cache: dict = {}
    skipped_fonts: set = set()

    for label, ch in tqdm(charset, desc=f'  Rendering {script} (context)'):
        out_dir = Path(script_dir) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cp = ord(ch)

        for font_path, family, style, cmap_set in font_meta:
            if cp not in cmap_set:
                continue

            if font_path not in font_cache:
                try:
                    font_cache[font_path] = ImageFont.truetype(font_path, RENDER_SIZE_PT)
                except Exception:
                    skipped_fonts.add(font_path)
                    font_cache[font_path] = None
            font = font_cache[font_path]
            if font is None:
                continue

            for dark in (False, True):
                mode = 'dark' if dark else 'light'
                for variant in range(variants_per_slot):
                    fname = f'{family}-{style}_{mode}_v{variant}.png'
                    fpath = out_dir / fname
                    if not update and fpath.exists():
                        continue

                    # zlib.crc32, not builtin hash() -- str hashing is
                    # randomized per-process (PYTHONHASHSEED) unless fixed,
                    # which would silently break the reproducibility this
                    # is meant to give (--update reruns must be idempotent).
                    key = f'{script}|{label}|{font_path}|{mode}|{variant}|{seed}'
                    rng = random.Random(zlib.crc32(key.encode('utf-8')))
                    tile = render_context_tile(font, cmap_set, ch, neighbor_chars, rng, dark,
                                               tile_size=tile_size)
                    if tile is not None:
                        tile.save(fpath)

    if skipped_fonts:
        print(f'  [{script}] {len(skipped_fonts)} font(s) failed to load (truetype error)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Render context-cropped character dataset (string-render + target-glyph crop).'
    )
    p.add_argument('--update', '-u', action='store_true', help='Re-render files that already exist')
    p.add_argument('--scripts', nargs='+', default=['latin'], choices=list(SCRIPT_NAMES),
                   help='Scripts to render. "all" expands to latin kana hangul cjk')
    p.add_argument('--charset', default=None, choices=list(CHARSETS.keys()),
                   help='(Latin only, backward-compat) alpha | extended')
    p.add_argument('--variants-per-slot', type=int, default=4,
                   help='Context-crop samples per (font, mode) slot (default 4). '
                        'Not capped to match char-dataset/\'s size -- more variants add '
                        'real diversity here (different neighbor chars + crop windows '
                        'each time), unlike the old pipeline where more copies of a '
                        'centered glyph would be near-duplicates.')
    p.add_argument('--hangul-top', type=int, default=500, metavar='N')
    p.add_argument('--cjk-top', type=int, default=3000, metavar='N')
    p.add_argument('--extra-fonts-dir', nargs='+', default=[], metavar='DIR')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tile-size', type=int, default=TILE_SIZE,
                   help=f'On-disk tile size in pixels (default {TILE_SIZE}). PNG byte size '
                        'scales with pixel count for this content -- 96px is ~25% smaller, '
                        '64px ~52% smaller than 128px (measured, not theoretical). Both old '
                        'and new pipelines get upsampled to the model\'s 224 input at train '
                        'time regardless of on-disk size, so shrinking this trades disk/'
                        'transfer size for a smaller pre-upsample source, not model input size.')
    p.add_argument('--dataset-name', default='char-dataset-ctx',
                   help='Output dataset folder name under Datasets/ (default: char-dataset-ctx). '
                        'Use a distinct name (e.g. char-dataset-ctx-small) to generate a '
                        'second variant without touching an existing one.')
    args = p.parse_args()
    out_root = str(_HERE / args.dataset_name)

    scripts = args.scripts
    if 'all' in scripts:
        scripts = ['latin', 'kana', 'hangul', 'cjk']

    jobs: list[tuple[str, list]] = []
    for script in scripts:
        if script == 'latin':
            key = args.charset or 'alpha'
            jobs.append(('latin', CHARSETS[key]))
        elif script == 'kana':
            jobs.append(('kana', _build_kana()))
        elif script == 'hangul':
            jobs.append(('hangul', _build_hangul(args.hangul_top)))
        elif script == 'cjk':
            jobs.append(('cjk', _build_cjk(args.cjk_top)))

    print(f'[render_chars_context] scripts={scripts}  render_pt={RENDER_SIZE_PT}  '
          f'variants_per_slot={args.variants_per_slot}  tile_size={args.tile_size}  '
          f'out={out_root}  update={args.update}')

    fonts_root = copy_system_fonts()
    import os
    font_files = [
        os.path.join(fonts_root, f) for f in os.listdir(fonts_root)
        if f.lower().endswith(('.ttf', '.otf', '.ttc'))
    ]
    if args.extra_fonts_dir:
        font_files += collect_extra_fonts(args.extra_fonts_dir)
    print(f'[render_chars_context] {len(font_files)} font files found')

    print('Scanning font cmaps (once) ...')
    font_meta = []
    for fp in tqdm(font_files, desc='  Loading fonts', leave=False):
        fam, sty, cmap = extract_cmap(fp)
        font_meta.append((fp, fam, sty, cmap))

    for script, charset in jobs:
        print(f'\n[render_chars_context] [{script}] {len(charset)} classes')
        render_charset_context(font_meta, charset, args.update, script,
                               args.variants_per_slot, args.seed, out_root, args.tile_size)

        script_dir = Path(out_root) / script
        total = sum(
            len(list((script_dir / label).glob('*.png')))
            for label, _ in charset if (script_dir / label).exists()
        )
        print(f'[render_chars_context] [{script}] done -- {total} images')

    print('\n[render_chars_context] All scripts complete.')


if __name__ == '__main__':
    main()
