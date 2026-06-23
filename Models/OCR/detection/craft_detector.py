"""
craft_detector.py — CRAFT text detection wrapper for the OCR pipeline.

Detects text regions in an image and returns bounding boxes or cropped PIL
images ready to feed into the char classifier or EasyOCR recognizer.

Usage
-----
    from detection.craft_detector import detect, detect_and_crop

    # Bounding boxes only
    boxes = detect("screenshot.png")
    # → list of (x1, y1, x2, y2) tuples in pixel coordinates

    # Cropped regions as PIL Images
    regions = detect_and_crop("screenshot.png")
    # → list of PIL.Image objects, one per detected text region

    # Control sensitivity
    boxes = detect("screenshot.png", text_threshold=0.5, link_threshold=0.3)

Models are downloaded once to ~/.craft_text_detector/ on first use.

Run from OCR/:
    python -m detection.craft_detector --image path/to/image.png
    python -m detection.craft_detector --image path/to/image.png --show-boxes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy model cache — loaded once per process
# ---------------------------------------------------------------------------
_craft_net   = None
_refine_net  = None
_model_cuda  = False


def _load_models(cuda: bool = False) -> tuple:
    global _craft_net, _refine_net, _model_cuda
    import craft_text_detector as craft

    if _craft_net is None or _model_cuda != cuda:
        _craft_net   = craft.load_craftnet_model(cuda=cuda)
        _refine_net  = craft.load_refinenet_model(cuda=cuda)
        _model_cuda  = cuda

    return _craft_net, _refine_net


# ---------------------------------------------------------------------------
# Source normalisation
# ---------------------------------------------------------------------------

def _to_numpy(source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Convert any supported input to an HxWx3 uint8 numpy array."""
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, Image.Image):
        return np.array(source.convert('RGB'))
    if isinstance(source, bytes):
        from io import BytesIO
        return np.array(Image.open(BytesIO(source)).convert('RGB'))
    # Path or str — let craft handle URL vs local transparently
    import craft_text_detector.image_utils as iu
    return iu.read_image(str(source))


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _poly_to_rect(poly: np.ndarray) -> tuple[int, int, int, int]:
    """Convert an Nx2 polygon to an axis-aligned (x1, y1, x2, y2) rect."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _clip_box(box: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float       = 0.4,
    long_size: int        = 1280,
    cuda: bool            = False,
    poly: bool            = False,
) -> list[tuple[int, int, int, int]]:
    """
    Detect text regions and return axis-aligned bounding boxes.

    Parameters
    ----------
    source          : image as path/str, bytes, numpy array, or PIL Image
    text_threshold  : character confidence threshold (lower = more detections)
    link_threshold  : link confidence threshold (lower = joins more characters)
    low_text        : text score lower bound (lower = picks up faint text)
    long_size       : longest side resized to this before inference
    cuda            : use GPU inference
    poly            : polygon grouping (True = curved text; broken on NumPy ≥1.24,
                      leave False unless using an older NumPy)

    Returns
    -------
    List of (x1, y1, x2, y2) bounding boxes in pixel coordinates,
    sorted top-to-bottom then left-to-right.
    """
    import craft_text_detector as craft

    image_np          = _to_numpy(source)
    craft_net, refine = _load_models(cuda)

    prediction = craft.get_prediction(
        image        = image_np,
        craft_net    = craft_net,
        refine_net   = refine,
        text_threshold = text_threshold,
        link_threshold = link_threshold,
        low_text     = low_text,
        cuda         = cuda,
        long_size    = long_size,
        poly         = poly,
    )

    h, w = image_np.shape[:2]
    boxes = []
    for poly_pts in prediction['boxes']:
        pts = np.array(poly_pts).reshape(-1, 2)
        box = _clip_box(_poly_to_rect(pts), w, h)
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            boxes.append(box)

    # Sort top-to-bottom, left-to-right
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def detect_and_crop(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float       = 0.4,
    long_size: int        = 1280,
    cuda: bool            = False,
    poly: bool            = False,
    pad: int              = 4,
) -> list[Image.Image]:
    """
    Detect text regions and return cropped PIL Images.

    Parameters
    ----------
    pad : extra pixels to expand each box on all sides before cropping

    Returns
    -------
    List of PIL Images (RGB), one per detected region, sorted top-to-bottom
    then left-to-right.  Feed directly into char_classifier.predict or
    EasyOCR's recognizer.
    """
    image_np = _to_numpy(source)
    pil      = Image.fromarray(image_np)
    h, w     = image_np.shape[:2]

    boxes = detect(
        source, text_threshold, link_threshold,
        low_text, long_size, cuda, poly,
    )

    crops = []
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crops.append(pil.crop((x1, y1, x2, y2)))

    return crops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _draw_boxes(image_np: np.ndarray, boxes: list) -> Image.Image:
    import cv2
    out = image_np.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return Image.fromarray(out)


def main():
    p = argparse.ArgumentParser(description='CRAFT text detection')
    p.add_argument('--image',           required=True, help='Input image path')
    p.add_argument('--text-threshold',  type=float, default=0.7)
    p.add_argument('--link-threshold',  type=float, default=0.4)
    p.add_argument('--low-text',        type=float, default=0.4)
    p.add_argument('--long-size',       type=int,   default=1280)
    p.add_argument('--no-refiner',      action='store_true',
                   help='Skip refinement network (faster, slightly less accurate)')
    p.add_argument('--cuda',            action='store_true')
    p.add_argument('--show-boxes',      action='store_true',
                   help='Save annotated image alongside input')
    args = p.parse_args()

    print(f'[craft] Detecting text in: {args.image}')
    image_np = _to_numpy(args.image)

    boxes = detect(
        source          = image_np,
        text_threshold  = args.text_threshold,
        link_threshold  = args.link_threshold,
        low_text        = args.low_text,
        long_size       = args.long_size,
        cuda            = args.cuda,
    )

    print(f'[craft] {len(boxes)} region(s) detected')
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        print(f'  {i:3d}. ({x1:4d},{y1:4d}) → ({x2:4d},{y2:4d})  '
              f'{x2-x1}×{y2-y1}px')

    if args.show_boxes:
        src    = Path(args.image)
        out    = src.with_name(src.stem + '_craft_boxes' + src.suffix)
        annotated = _draw_boxes(image_np, boxes)
        annotated.save(out)
        print(f'[craft] Annotated image saved: {out}')


if __name__ == '__main__':
    main()
