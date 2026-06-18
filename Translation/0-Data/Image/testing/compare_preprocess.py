"""Preprocessing comparison: baseline vs enhanced vs discord-aware vs experimental.

Runs all preprocessing variants on every image in 0-Data/Image/data/, OCRs each
with the zh reader, and prints a side-by-side confidence/text report. Saves
processed images to 0-Data/Image/data/preprocess_comparison/ for visual inspection.

Usage:
    python compare_preprocess.py
    python compare_preprocess.py --images path/to/dir
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 so CJK characters print correctly on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

HERE = Path(__file__).parent

# ocr.py lives in Translation/2-Image/ — three levels up from testing/
sys.path.insert(0, str(HERE.parent.parent.parent / "2-Image"))

from ocr import (
    preprocess,
    preprocess_enhanced,
    preprocess_discord,
    preprocess_otsu,
    preprocess_light_denoise,
    preprocess_bilateral,
    load_image_from_path,
    _get_reader_zh,
    _split_merged_words,
)

READ_KWARGS = dict(
    width_ths=1e4,
    add_margin=0.1,
    low_text=0.1,
    text_threshold=0.8,
    paragraph=False,
)

VARIANTS: list[tuple[str, object]] = [
    ("baseline",      preprocess),
    ("enhanced",      preprocess_enhanced),
    ("discord",       preprocess_discord),
    ("otsu",          preprocess_otsu),
    ("light_denoise", preprocess_light_denoise),
    ("bilateral",     preprocess_bilateral),
]


def _run_ocr(processed: np.ndarray) -> tuple[list[tuple[str, float]], float]:
    raw = _get_reader_zh().readtext(processed, **READ_KWARGS)
    segments = [
        (_split_merged_words(t.strip()), round(c, 4))
        for _, t, c in raw
        if t.strip()
    ]
    avg = round(sum(c for _, c in segments) / len(segments), 4) if segments else 0.0
    return segments, avg


def _conf_bar(v: float, width: int = 14) -> str:
    filled = round(v * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {v*100:5.1f}%"


def compare_image(path: Path, out_dir: Path) -> None:
    print(f"\n{'='*72}")
    print(f"  {path.name}")
    print(f"{'='*72}")

    raw = load_image_from_path(path)

    results: dict[str, tuple[list, float, np.ndarray]] = {}
    for name, fn in VARIANTS:
        processed = fn(raw)
        cv2.imwrite(str(out_dir / f"{path.stem}_{name}.png"), processed)
        segs, avg = _run_ocr(processed)
        results[name] = (segs, avg, processed)

    col = 16
    print(f"\n  {'':12s}", end="")
    for name, _ in VARIANTS:
        print(f"  {name.upper():<{col}}", end="")
    print()

    print(f"  {'':12s}", end="")
    for _ in VARIANTS:
        print(f"  {'-'*col}", end="")
    print()

    print(f"  {'Avg conf':<12s}", end="")
    for name, _ in VARIANTS:
        _, avg, _ = results[name]
        print(f"  {_conf_bar(avg):<{col}}", end="")
    print()

    print(f"  {'Tokens':<12s}", end="")
    for name, _ in VARIANTS:
        segs, _, _ = results[name]
        print(f"  {len(segs):<{col}}", end="")
    print()

    _, base_avg, _ = results["baseline"]
    print(f"  {'vs baseline':<12s}", end="")
    print(f"  {'---':<{col}}", end="")
    for name, _ in VARIANTS[1:]:
        _, avg, _ = results[name]
        delta = avg - base_avg
        sign = "+" if delta >= 0 else ""
        print(f"  {sign}{delta*100:.1f}%{'':<{col-7}}", end="")
    print()

    print()
    for name, _ in VARIANTS:
        segs, avg, _ = results[name]
        print(f"  --- {name.upper()} ---")
        for text, conf in segs:
            print(f"    ({conf*100:.0f}%) {text}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=HERE.parent / "data",
        help="Directory of test images (default: 0-Data/Image/data/)",
    )
    args = parser.parse_args()

    image_dir: Path = args.images
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    out_dir = HERE.parent / "data" / "preprocess_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EasyOCR reader (first run may take a moment)...")
    _get_reader_zh()

    print(f"\nComparing {len(images)} image(s) | variants: {[n for n,_ in VARIANTS]}")
    print(f"Preprocessed output -> {out_dir}")

    for img_path in images:
        compare_image(img_path, out_dir)

    print(f"\n{'='*72}")
    print(f"Done. Check {out_dir} for visual diffs.")


if __name__ == "__main__":
    main()
