import sys, math, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "OCR"))
from grid_augments import (TileGrid3x3, TileGrid3x3Rotated,
                            TileGrid3x3Pair, TileGrid3x3PairRotated,
                            TileGrid3x3Orbital, TileGrid3x3OrbitalRotated,
                            _GRID_ANGLES)

DATASET = Path(__file__).parent.parent.parent / "Datasets" / "char-dataset-legacy" / "latin"
CHARS   = ["cap_A", "low_b", "low_d", "dig_6", "dig_9", "cap_G", "low_p", "low_q"]
random.seed(42)

try:
    label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",   11)
    hdr_font   = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 11)
    ang_font   = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",   10)
except Exception:
    label_font = hdr_font = ang_font = ImageFont.load_default()

# ------------------------------------------------------------------
# Peer pools (light-mode tiles, grouped by script)
# ------------------------------------------------------------------
latin_pool = []
digit_pool = []
for cls_dir in sorted(DATASET.iterdir()):
    if not cls_dir.is_dir():
        continue
    tiles = sorted(cls_dir.glob("*_96_light.png"))
    if not tiles:
        continue
    img = Image.open(tiles[0]).convert("RGB").resize((128, 128))
    (digit_pool if cls_dir.name.startswith("dig_") else latin_pool).append(img)

def pool_for(label: str) -> list:
    return digit_pool if label.startswith("dig_") else latin_pool

# Pre-load focus tiles
focus_tiles = {}
for label in CHARS:
    tiles = sorted((DATASET / label).glob("*_96_light.png"))
    focus_tiles[label] = Image.open(tiles[0]).convert("RGB").resize((128, 128))

# ------------------------------------------------------------------
# Build one base grid per char for each variant (same peer draw reused
# across all 8 rotation angles so the strip is consistent per row).
# ------------------------------------------------------------------
base_grids     = {}   # focus char only
single_grids   = {}   # focus + random peers
pair_grids     = {}   # 2-char pairs + random peers
orbital_grids  = {}   # focus char rotated in each surrounding cell

for label in CHARS:
    tile = focus_tiles[label]
    pool = pool_for(label)
    base_grids[label]    = (tile, TileGrid3x3()(tile))
    single_grids[label]  = (tile, TileGrid3x3(peer_pool=pool)(tile))
    pair_grids[label]    = (tile, TileGrid3x3Pair(peer_pool=pool)(tile))
    orbital_grids[label] = (tile, TileGrid3x3Orbital()(tile))

# ------------------------------------------------------------------
# Panel builder: 8-char rows × 8-angle columns
# ------------------------------------------------------------------
LABEL_W = 80
GAP     = 8
HDR_H   = 32
ANG_H   = 16
ROW_PAD = 10

def make_rotated(base, angle, bg_color, canvas_w, canvas_h):
    rotated = base.rotate(-angle, expand=True, fillcolor=bg_color)
    canvas  = Image.new(base.mode, (canvas_w, canvas_h), bg_color)
    canvas.paste(rotated, ((canvas_w - rotated.width)  // 2,
                            (canvas_h - rotated.height) // 2))
    return canvas

def build_panel(grids: dict, title: str) -> Image.Image:
    # Universal max canvas size across all chars and all angles
    max_cell_w = max_cell_h = 0
    for label, (tile, base) in grids.items():
        gw, gh = base.size
        for a in _GRID_ANGLES:
            rad = math.radians(a)
            w = int(math.ceil(abs(gw * math.cos(rad)) + abs(gh * math.sin(rad))))
            h = int(math.ceil(abs(gw * math.sin(rad)) + abs(gh * math.cos(rad))))
            max_cell_w = max(max_cell_w, w)
            max_cell_h = max(max_cell_h, h)

    cell_slot_w = max_cell_w + GAP
    cell_slot_h = max_cell_h + ANG_H + GAP
    row_h       = cell_slot_h + ROW_PAD
    total_w     = LABEL_W + len(_GRID_ANGLES) * cell_slot_w + GAP
    total_h     = HDR_H + len(CHARS) * row_h

    panel = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw  = ImageDraw.Draw(panel)

    for c, angle in enumerate(_GRID_ANGLES):
        x = LABEL_W + c * cell_slot_w + max_cell_w // 2 - 10
        draw.text((x,      4), f"{angle}°",  fill=(60,  60,  60),  font=hdr_font)
        draw.text((x + 2, 18), "rotation",   fill=(100, 100, 100),  font=ang_font)

    for r, label in enumerate(CHARS):
        y  = HDR_H + r * row_h
        draw.text((4, y + cell_slot_h // 2 - 6), label, fill=(40, 40, 40), font=label_font)

        tile, base = grids[label]
        bg = tile.getpixel((0, 0))

        for c, angle in enumerate(_GRID_ANGLES):
            cell = make_rotated(base, angle, bg, max_cell_w, max_cell_h)
            panel.paste(cell, (LABEL_W + c * cell_slot_w, y))

        draw.line([(0, y + row_h - ROW_PAD // 2), (total_w, y + row_h - ROW_PAD // 2)],
                  fill=(210, 210, 210))

    return panel

# ------------------------------------------------------------------
# Generate and save both panels
# ------------------------------------------------------------------
here = Path(__file__).parent.parent.parent / "Datasets"

p0 = build_panel(base_grids, "TileGrid3x3 (base)")
out0 = here / "sample_tilegrid_base.png"
p0.save(out0)
print(f"Saved: {out0}  ({p0.width}x{p0.height}px)")

p1 = build_panel(single_grids, "TileGrid3x3")
out1 = here / "sample_tilegrid.png"
p1.save(out1)
print(f"Saved: {out1}  ({p1.width}x{p1.height}px)")

p2 = build_panel(pair_grids, "TileGrid3x3Pair")
out2 = here / "sample_tilegrid_pair.png"
p2.save(out2)
print(f"Saved: {out2}  ({p2.width}x{p2.height}px)")

p3 = build_panel(orbital_grids, "TileGrid3x3Orbital")
out3 = here / "sample_tilegrid_orbital.png"
p3.save(out3)
print(f"Saved: {out3}  ({p3.width}x{p3.height}px)")
