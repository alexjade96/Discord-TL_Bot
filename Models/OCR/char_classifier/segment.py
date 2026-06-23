"""
segment.py — Column-projection character segmentation.

Splits a word/line crop (PIL Image) into individual character crops by finding
ink-valley cut points in the horizontal column projection.

Usage:
    from char_classifier.segment import split_into_chars
    chars = split_into_chars(word_crop)   # list[PIL.Image]
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _column_projection(gray: np.ndarray) -> np.ndarray:
    """Ink density per column: high where characters are, low in gaps."""
    inv = 255 - gray.astype(np.float32)  # invert so ink = high value
    return inv.sum(axis=0)


def _smooth(proj: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """
    Gaussian smoothing via scipy (optional; sigma=0 skips smoothing).

    For clean typeset text, smoothing fills inter-character gaps and prevents
    segmentation.  Leave sigma=0.0 (the default) for typeset images.  Pass a
    small positive value (e.g. 0.5) only for heavily degraded or handwritten input.
    """
    if sigma < 0.01:
        return proj.copy()
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(proj, sigma)
    except ImportError:
        k = max(1, int(sigma * 2))
        kernel = np.ones(k) / k
        return np.convolve(proj, kernel, mode='same')


def _find_cuts(
    proj: np.ndarray,
    threshold_frac: float = 0.08,
    min_gap: int = 2,
) -> list[int]:
    """
    Find column midpoints where a vertical cut can be placed.

    A cut is placed at the midpoint of each run of columns whose projection
    is below threshold_frac * max(proj) and at least min_gap columns wide.
    """
    if proj.max() == 0:
        return []
    threshold = threshold_frac * proj.max()
    below = proj < threshold

    cuts: list[int] = []
    in_gap, gap_start = False, 0
    for i, b in enumerate(below):
        if b and not in_gap:
            in_gap, gap_start = True, i
        elif not b and in_gap:
            in_gap = False
            if (i - gap_start) >= min_gap:
                cuts.append((gap_start + i) // 2)
    if in_gap and (len(below) - gap_start) >= min_gap:
        cuts.append((gap_start + len(below)) // 2)
    return cuts


def split_into_chars(
    crop: Image.Image,
    min_char_width: int   = 4,
    pad:            int   = 1,
    threshold_frac: float = 0.08,
    sigma:          float = 0.0,
) -> list[Image.Image]:
    """
    Split a word/line crop into individual character crops using column projection.

    Parameters
    ----------
    crop            : PIL Image of a word or character sequence
    min_char_width  : discard sub-crops narrower than this many pixels
    pad             : extra pixels added to the left/right edge of each char crop
    threshold_frac  : ink density fraction below which a column is treated as a gap
    sigma           : Gaussian smoothing sigma for the projection curve

    Returns
    -------
    List of PIL Images, one per detected character.  Returns ``[crop]`` unchanged
    when no gap is found (single-character input or unsegmentable crop).
    """
    gray = np.array(crop.convert('L'))
    proj = _column_projection(gray)
    proj = _smooth(proj, sigma)
    cuts = _find_cuts(proj, threshold_frac, min_gap=2)

    w, h = crop.size
    boundaries = [0] + cuts + [w]

    chars: list[Image.Image] = []
    for i in range(len(boundaries) - 1):
        x1 = max(0, boundaries[i] - pad)
        x2 = min(w, boundaries[i + 1] + pad)
        if x2 - x1 < min_char_width:
            continue
        chars.append(crop.crop((x1, 0, x2, h)))

    return chars if chars else [crop]
