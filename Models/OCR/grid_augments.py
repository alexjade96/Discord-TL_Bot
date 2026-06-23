"""
grid_augments.py — 3×3 grid augmentation transforms for OCR character training.

All transforms accept a single PIL Image (one rendered character tile) and return
a PIL Image showing the character arranged in a 3×3 grid of cells.  They are
designed to be composed with torchvision transforms in char_classifier/data.py
and are importable as OCR-level utilities from this shared location.

Public classes
--------------
TileGrid3x3            — 3×3 grid; surrounding cells draw from optional peer pool
TileGrid3x3Rotated     — same grid, full-grid rotation at a random angle
TileGrid3x3Pair        — each cell holds two glyphs side by side
TileGrid3x3PairRotated — same pair grid with full-grid rotation
TileGrid3x3Orbital     — focus char in all cells; surrounds rotated clockwise
TileGrid3x3OrbitalRotated — orbital grid with full-grid rotation
"""

import math
import random

import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _crop_glyph(img: Image.Image) -> tuple[Image.Image, bool]:
    """Return (tight-cropped glyph, is_light_background)."""
    gray     = np.array(img.convert('L'))
    is_light = int(gray[0, 0]) > 128
    thr      = 200 if is_light else 50
    cmp      = (gray < thr) if is_light else (gray > thr)
    mask     = Image.fromarray(cmp.astype(np.uint8) * 255, 'L')
    bbox     = mask.getbbox() or (0, 0, img.width, img.height)
    return img.crop(bbox), is_light


def _scale_peer(g: Image.Image, target_h: int, center_light: bool,
                mode: str) -> Image.Image:
    """Invert bg if needed and scale peer glyph to target_h."""
    _, peer_light = _crop_glyph(g)
    if peer_light != center_light:
        g = ImageOps.invert(g.convert('RGB')).convert(mode)
    scale = target_h / g.height if g.height > 0 else 1.0
    return g.resize((max(1, int(g.width * scale)), target_h), Image.LANCZOS)


def _build_grid(cells: list, ch: int, h_gap: int, v_gap: int,
                padding: int, bg_color, mode: str) -> Image.Image:
    """Paste a flat list of 9 cell images into a 3×3 grid."""
    col_w   = [max(cells[r * 3 + c].width for r in range(3)) for c in range(3)]
    p       = padding
    total_w = sum(col_w) + h_gap * 2 + p * 2
    total_h = ch * 3 + v_gap * 2 + p * 2
    grid    = Image.new(mode, (total_w, total_h), bg_color)
    col_x   = [p,
               p + col_w[0] + h_gap,
               p + col_w[0] + h_gap + col_w[1] + h_gap]
    for row in range(3):
        for col in range(3):
            cell = cells[row * 3 + col]
            x = col_x[col] + (col_w[col] - cell.width) // 2
            y = p + row * (ch + v_gap)
            grid.paste(cell, (x, y))
    return grid


def _rotate_grid(grid: Image.Image, angle: int,
                 bg_color, mode: str,
                 canvas_w: int, canvas_h: int) -> Image.Image:
    """Rotate grid and centre on a pre-sized canvas."""
    rotated = grid.rotate(-angle, expand=True, fillcolor=bg_color)
    canvas  = Image.new(mode, (canvas_w, canvas_h), bg_color)
    canvas.paste(rotated, ((canvas_w - rotated.width)  // 2,
                            (canvas_h - rotated.height) // 2))
    return canvas


_GRID_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def _max_rotated_canvas(w: int, h: int) -> tuple[int, int]:
    """Largest bounding box across all 8 rotation angles."""
    max_w = max_h = 0
    for a in _GRID_ANGLES:
        rad   = math.radians(a)
        max_w = max(max_w, int(math.ceil(abs(w * math.cos(rad)) + abs(h * math.sin(rad)))))
        max_h = max(max_h, int(math.ceil(abs(w * math.sin(rad)) + abs(h * math.cos(rad)))))
    return max_w, max_h


# ---------------------------------------------------------------------------
# Grid transforms
# ---------------------------------------------------------------------------

class TileGrid3x3:
    """
    Crop each glyph to its tight bounding box, then tile it into a 3×3 grid
    with minimal inter-glyph spacing — mimicking sequential character placement
    in a word or sentence rather than isolated padded tiles.

    h_gap:     horizontal pixels between columns (kerning approximation).
    v_gap:     vertical pixels between rows (leading approximation).
    peer_pool: optional list of PIL Images to draw surrounding cells from.
               If None, all 9 cells repeat the centre character.
               Peers are scaled to match the centre glyph height; bg colour is
               inverted when a peer's background differs from the centre's.
    """

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 peer_pool: list | None = None):
        self.h_gap     = h_gap
        self.v_gap     = v_gap
        self.padding   = padding
        self.peer_pool = peer_pool

    def __call__(self, img: Image.Image) -> Image.Image:
        center, center_light = _crop_glyph(img)
        ch       = center.height
        bg_color = img.getpixel((0, 0))

        if self.peer_pool:
            peers = [_scale_peer(_crop_glyph(raw)[0], ch, center_light, img.mode)
                     for raw in random.choices(self.peer_pool, k=8)]
        else:
            peers = [center] * 8

        cells = peers[:4] + [center] + peers[4:]
        return _build_grid(cells, ch, self.h_gap, self.v_gap,
                           self.padding, bg_color, img.mode)


class TileGrid3x3Rotated:
    """
    Builds the same tight 3×3 grid as TileGrid3x3, then rotates the ENTIRE
    grid by a randomly chosen angle from 8 evenly-spaced orientations
    (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°), simulating a full line of
    text captured at an angle or skew rather than rotating individual glyphs.
    Canvas expands to avoid clipping at diagonal angles.
    """

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 peer_pool: list | None = None):
        self._grid_tf = TileGrid3x3(h_gap=h_gap, v_gap=v_gap, padding=padding,
                                    peer_pool=peer_pool)

    def __call__(self, img: Image.Image) -> Image.Image:
        grid         = self._grid_tf(img)
        bg_color     = img.getpixel((0, 0))
        max_w, max_h = _max_rotated_canvas(*grid.size)
        angle        = random.choice(_GRID_ANGLES)
        return _rotate_grid(grid, angle, bg_color, img.mode, max_w, max_h)


class TileGrid3x3Pair:
    """
    Like TileGrid3x3 but each cell holds two glyphs side by side — the focus
    character appears twice in the centre cell; surrounding cells get two
    randomly sampled peers (or two copies of the focus when peer_pool is None).
    Inter-cell spacing is identical to TileGrid3x3.

    pair_gap: pixels between the two glyphs within a single cell.
    """

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 pair_gap: int = 2, peer_pool: list | None = None):
        self.h_gap     = h_gap
        self.v_gap     = v_gap
        self.padding   = padding
        self.pair_gap  = pair_gap
        self.peer_pool = peer_pool

    def _make_pair(self, g1: Image.Image, g2: Image.Image,
                   bg_color, mode: str) -> Image.Image:
        if g2.height != g1.height:
            scale = g1.height / g2.height if g2.height > 0 else 1.0
            g2 = g2.resize((max(1, int(g2.width * scale)), g1.height), Image.LANCZOS)
        pair = Image.new(mode, (g1.width + self.pair_gap + g2.width, g1.height), bg_color)
        pair.paste(g1, (0, 0))
        pair.paste(g2, (g1.width + self.pair_gap, 0))
        return pair

    def __call__(self, img: Image.Image) -> Image.Image:
        center, center_light = _crop_glyph(img)
        ch       = center.height
        bg_color = img.getpixel((0, 0))

        def peer():
            raw, _ = _crop_glyph(random.choice(self.peer_pool))
            return _scale_peer(raw, ch, center_light, img.mode)

        cells = []
        for i in range(9):
            if i == 4 or not self.peer_pool:
                cells.append(self._make_pair(center, center, bg_color, img.mode))
            else:
                cells.append(self._make_pair(peer(), peer(), bg_color, img.mode))

        return _build_grid(cells, ch, self.h_gap, self.v_gap,
                           self.padding, bg_color, img.mode)


class TileGrid3x3PairRotated:
    """TileGrid3x3Pair with full-grid rotation at one of 8 evenly-spaced angles."""

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 pair_gap: int = 2, peer_pool: list | None = None):
        self._grid_tf = TileGrid3x3Pair(h_gap=h_gap, v_gap=v_gap, padding=padding,
                                        pair_gap=pair_gap, peer_pool=peer_pool)

    def __call__(self, img: Image.Image) -> Image.Image:
        grid         = self._grid_tf(img)
        bg_color     = img.getpixel((0, 0))
        max_w, max_h = _max_rotated_canvas(*grid.size)
        angle        = random.choice(_GRID_ANGLES)
        return _rotate_grid(grid, angle, bg_color, img.mode, max_w, max_h)


class TileGrid3x3Orbital:
    """
    3×3 grid where the focus character fills every cell.  The centre is
    unrotated; each surrounding cell shows the character rotated at one of
    the 8 evenly-spaced angles, arranged clockwise from the top-left:

        [  0°][  45°][ 90°]
        [315°][focus][135°]
        [270°][ 225°][180°]

    All cells are padded to the same size (max bounding box across all
    rotations) so the grid is uniform regardless of character aspect ratio.
    """
    _SURROUND = [0, 1, 2, 5, 8, 7, 6, 3]   # clockwise from top-left, skip centre
    _ANGLES   = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6):
        self.h_gap   = h_gap
        self.v_gap   = v_gap
        self.padding = padding

    def _rotated_cell(self, glyph: Image.Image, angle: int,
                      cell_w: int, cell_h: int, bg_color, mode: str) -> Image.Image:
        rot    = glyph.rotate(-angle, expand=True, fillcolor=bg_color)
        canvas = Image.new(mode, (cell_w, cell_h), bg_color)
        canvas.paste(rot, ((cell_w - rot.width) // 2, (cell_h - rot.height) // 2))
        return canvas

    def __call__(self, img: Image.Image) -> Image.Image:
        glyph, _ = _crop_glyph(img)
        gw, gh   = glyph.size
        bg_color = img.getpixel((0, 0))

        cell_w = cell_h = 0
        for a in self._ANGLES:
            rad    = math.radians(a)
            cell_w = max(cell_w, int(math.ceil(abs(gw * math.cos(rad)) + abs(gh * math.sin(rad)))))
            cell_h = max(cell_h, int(math.ceil(abs(gw * math.sin(rad)) + abs(gh * math.cos(rad)))))

        cells    = [None] * 9
        cells[4] = self._rotated_cell(glyph, 0, cell_w, cell_h, bg_color, img.mode)
        for pos, angle in zip(self._SURROUND, self._ANGLES):
            cells[pos] = self._rotated_cell(glyph, angle, cell_w, cell_h, bg_color, img.mode)

        return _build_grid(cells, cell_h, self.h_gap, self.v_gap,
                           self.padding, bg_color, img.mode)


class TileGrid3x3OrbitalRotated:
    """TileGrid3x3Orbital with full-grid rotation at one of 8 evenly-spaced angles."""

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6):
        self._grid_tf = TileGrid3x3Orbital(h_gap=h_gap, v_gap=v_gap, padding=padding)

    def __call__(self, img: Image.Image) -> Image.Image:
        grid         = self._grid_tf(img)
        bg_color     = img.getpixel((0, 0))
        max_w, max_h = _max_rotated_canvas(*grid.size)
        angle        = random.choice(_GRID_ANGLES)
        return _rotate_grid(grid, angle, bg_color, img.mode, max_w, max_h)


class RandomGridAugment:
    """
    Randomly selects one grid augmentation per call from a provided list.

    Exposes all grid variants within a single training run so each sample
    can be presented in a different grid context.  Pass the full list of
    instantiated transforms; one is chosen uniformly at random each time.
    """

    def __init__(self, augments: list):
        self.augments = augments

    def __call__(self, img: Image.Image) -> Image.Image:
        return random.choice(self.augments)(img)
