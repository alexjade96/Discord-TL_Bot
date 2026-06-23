"""
ocr_pipeline.py — CRAFT detection + script-routed recognition.

Script routing per detected region:
  Hangul   → EasyOCR Korean reader  (easyocr, already in requirements)
  CJK/Kana → manga-ocr              (pip install manga-ocr)
  Latin    → EasyOCR English        (word/line crops)
           → char_classifier        (single-char crops, when checkpoint present)

manga-ocr output is checked post-hoc: if Hangul appears in the result the crop
is re-run through EasyOCR Korean, which is better calibrated for that script.

char_classifier is not used in the main recognize() pipeline (CRAFT returns
word-level boxes; char_classifier expects single isolated characters). It is
exposed as recognize_char() for use after a character segmentation step.

Run from OCR/:
    python -m ocr_pipeline --image screenshot.png
    python -m ocr_pipeline --image screenshot.png --show-boxes --save-crops
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from detection.craft_detector import detect, detect_and_crop

# ---------------------------------------------------------------------------
# Script detection patterns (Unicode ranges, checked on decoded text strings)
# ---------------------------------------------------------------------------
_HANGUL_RE = re.compile(r'[가-힣ᄀ-ᇿ㄰-㆏]')
_KANA_RE   = re.compile(r'[぀-ヿ]')
_CJK_RE    = re.compile(r'[　-鿿豈-﫿]')
_LATIN_RE  = re.compile(r'[A-Za-zÀ-ɏ]')

_DEFAULT_CHAR_CKPT = str(Path(__file__).parent / 'checkpoints' / 'best.pt')

# ---------------------------------------------------------------------------
# Lazy-loaded model handles
# ---------------------------------------------------------------------------
_manga_ocr_model = None
_easyocr_ko      = None
_easyocr_en      = None
_char_clf        = None
_char_clf_names  = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Recognition:
    """Result for a single crop or text region."""
    text:       str
    confidence: float
    script:     str    # 'latin' | 'cjk' | 'hangul' | 'unknown'
    method:     str    # 'manga-ocr' | 'easyocr-ko' | 'easyocr-en' | 'char-clf'


@dataclass
class RegionResult:
    """Detection + recognition for one bounding box."""
    text:       str
    confidence: float
    script:     str
    method:     str
    bbox:       tuple  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Script utilities
# ---------------------------------------------------------------------------

def _text_script(text: str) -> str:
    """Classify the dominant script in a decoded string."""
    if _HANGUL_RE.search(text):
        return 'hangul'
    if _KANA_RE.search(text) or _CJK_RE.search(text):
        return 'cjk'
    if _LATIN_RE.search(text):
        return 'latin'
    return 'unknown'


def _latin_fraction(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if _LATIN_RE.match(c)) / len(alpha)


# Fullwidth Latin block: U+FF01–U+FF5E maps to U+0021–U+007E by subtracting 0xFEE0.
# manga-ocr returns fullwidth Latin when processing Latin text, because its training
# data (manga) uses fullwidth characters for stylistic Latin inside Japanese text.
def _normalize_fullwidth(text: str) -> str:
    """Convert fullwidth ASCII (U+FF01–U+FF5E) to standard halfwidth ASCII."""
    return ''.join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )


def _screen_script(crop: Image.Image) -> str:
    """
    Fast pixel-level script pre-screen.

    At Discord font sizes (16–22 px), CC geometry does not reliably distinguish
    Latin from CJK/Hangul: Hangul jamo strokes and Latin letters produce
    overlapping aspect-ratio distributions.  The correct approach requires a
    small trained script classifier; that is left as a future improvement.

    For v1 this function always returns 'cjk_or_hangul' so every crop runs
    through manga-ocr.  The fullwidth post-hoc check in recognize_crop()
    detects and re-routes Latin crops to EasyOCR English.  Korean crops are a
    known v1 limitation — manga-ocr maps them to Japanese (see docstring on
    recognize_crop).
    """
    return 'cjk_or_hangul'


# ---------------------------------------------------------------------------
# Model loaders (lazy, cached per process)
# ---------------------------------------------------------------------------

def _load_manga_ocr():
    global _manga_ocr_model
    if _manga_ocr_model is None:
        try:
            from manga_ocr import MangaOcr
            print('[ocr_pipeline] Loading manga-ocr ...')
            _manga_ocr_model = MangaOcr()
        except ImportError:
            raise ImportError(
                'manga-ocr is not installed. '
                'Run: pip install manga-ocr'
            )
    return _manga_ocr_model


def _load_easyocr_ko():
    global _easyocr_ko
    if _easyocr_ko is None:
        import easyocr
        print('[ocr_pipeline] Loading EasyOCR Korean reader ...')
        _easyocr_ko = easyocr.Reader(['ko', 'en'], gpu=False)
    return _easyocr_ko


def _load_easyocr_en():
    global _easyocr_en
    if _easyocr_en is None:
        import easyocr
        print('[ocr_pipeline] Loading EasyOCR English reader ...')
        _easyocr_en = easyocr.Reader(['en'], gpu=False)
    return _easyocr_en


def _load_char_clf(device=None):
    """Load char_classifier if checkpoint exists. Returns (model, class_names) or (None, None)."""
    global _char_clf, _char_clf_names
    if _char_clf is not None:
        return _char_clf, _char_clf_names

    ckpt = Path(_DEFAULT_CHAR_CKPT)
    names_file = ckpt.parent / 'class_names.json'
    if not ckpt.exists() or not names_file.exists():
        return None, None

    import json, torch
    from char_classifier.model_builder import create_model
    from char_classifier.utils import get_device, load_checkpoint

    device = device or get_device()
    names  = json.load(open(names_file))
    model  = create_model('dinov2_vits14', num_classes=len(names), freeze_base=False)
    model  = model.to(device)
    load_checkpoint(str(ckpt), model, device=device)
    model.eval()
    _char_clf       = model
    _char_clf_names = names
    print(f'[ocr_pipeline] char_classifier loaded ({len(names)} classes)')
    return _char_clf, _char_clf_names


# ---------------------------------------------------------------------------
# Per-script recognition functions
# ---------------------------------------------------------------------------

def _easyocr_flatten(results) -> tuple[str, float]:
    """Collapse EasyOCR list of (bbox, text, conf) into (text, avg_conf)."""
    if not results:
        return '', 0.0
    texts = [r[1] for r in results]
    confs = [float(r[2]) for r in results]
    return ' '.join(t for t in texts if t.strip()), round(sum(confs) / len(confs), 4)


def _run_manga_ocr(crop: Image.Image) -> tuple[str, float]:
    """Returns (text, confidence). manga-ocr does not expose a score; proxy 0.85."""
    ocr  = _load_manga_ocr()
    text = ocr(crop)
    return text, 0.85


def _run_easyocr_ko(crop: Image.Image) -> tuple[str, float]:
    reader = _load_easyocr_ko()
    return _easyocr_flatten(reader.readtext(np.array(crop)))


def _run_easyocr_en(crop: Image.Image) -> tuple[str, float]:
    reader = _load_easyocr_en()
    return _easyocr_flatten(reader.readtext(np.array(crop)))


# ---------------------------------------------------------------------------
# Public: single-crop recognition
# ---------------------------------------------------------------------------

def recognize_crop(
    crop: Image.Image,
    script_hint: Optional[str] = None,
) -> Recognition:
    """
    Recognize text in a single PIL Image crop.

    Parameters
    ----------
    crop        : PIL Image, e.g. from detect_and_crop()
    script_hint : 'latin' | 'cjk' | 'hangul' | None
                  If None, the script is inferred from pixel geometry then
                  confirmed from the decoded text.

    Returns
    -------
    Recognition dataclass with text, confidence, script, method.
    """
    crop = crop.convert('RGB')
    hint = script_hint or _screen_script(crop)

    if hint == 'latin':
        text, conf = _run_easyocr_en(crop)
        return Recognition(text, conf, 'latin', 'easyocr-en')

    if hint == 'hangul':
        text, conf = _run_easyocr_ko(crop)
        return Recognition(text, conf, 'hangul', 'easyocr-ko')

    # 'cjk_or_hangul' or 'unknown' — try manga-ocr first
    text, conf = _run_manga_ocr(crop)

    # manga-ocr returns fullwidth Latin (Ａ-Ｚ, ａ-ｚ) when the crop is really
    # Latin text, because it is trained on manga which uses fullwidth for stylistic
    # Latin.  Normalize and re-route to EasyOCR English for better accuracy.
    normalized = _normalize_fullwidth(text)
    if _latin_fraction(normalized) > 0.6:
        text, conf = _run_easyocr_en(crop)
        return Recognition(text, conf, 'latin', 'easyocr-en')

    # manga-ocr is trained on Japanese; it maps Korean to Japanese.  If the
    # normalized output contains Hangul we got lucky; re-route to EasyOCR Korean.
    # If it returns Japanese for what is actually Korean, we miss it here — known
    # limitation of v1; a dedicated Hangul pre-screen would fix this.
    script = _text_script(normalized)
    if script == 'hangul':
        text, conf = _run_easyocr_ko(crop)
        return Recognition(text, conf, 'hangul', 'easyocr-ko')

    text = normalized
    if not text.strip():
        script = 'unknown'

    return Recognition(text, conf, script or 'cjk', 'manga-ocr')


def recognize_char(
    crop: Image.Image,
    device=None,
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """
    Classify a single isolated character crop via char_classifier.

    Designed for use after a character segmentation step (column projection,
    connected component splitting) that produces per-character crops from a
    word-level CRAFT box.

    Returns a list of (label, confidence) tuples, length = top_k.
    Falls back to recognize_crop() if char_classifier checkpoint is absent.
    """
    import torch
    from torchvision import transforms

    clf, names = _load_char_clf(device)
    if clf is None:
        # No checkpoint — fall back to full-crop recognizer
        r = recognize_crop(crop)
        return [(r.text, r.confidence)]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(crop.convert('RGB')).unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    with torch.inference_mode():
        probs = torch.softmax(clf(tensor), dim=1)[0]

    k = min(top_k, len(names))
    values, indices = torch.topk(probs, k)
    return [(names[i.item()], round(v.item(), 4)) for i, v in zip(indices, values)]


# ---------------------------------------------------------------------------
# Public: full pipeline
# ---------------------------------------------------------------------------

def recognize(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text:       float = 0.4,
    long_size:      int   = 1280,
    pad:            int   = 4,
    cuda:           bool  = False,
) -> list[RegionResult]:
    """
    Full pipeline: CRAFT detection → script routing → recognition.

    Parameters
    ----------
    source          : image as path, bytes, numpy array, or PIL Image
    text_threshold  : CRAFT character confidence threshold
    link_threshold  : CRAFT link confidence threshold
    low_text        : CRAFT text score lower bound
    long_size       : CRAFT resize limit (longest edge)
    pad             : extra pixels to expand each crop on all sides
    cuda            : use GPU for CRAFT and EasyOCR (manga-ocr uses CPU by default)

    Returns
    -------
    List of RegionResult, one per detected text region, sorted top-to-bottom
    then left-to-right (CRAFT order is preserved).
    """
    if not isinstance(source, Image.Image):
        if isinstance(source, bytes):
            from io import BytesIO
            source = Image.open(BytesIO(source)).convert('RGB')
        else:
            source = Image.open(source).convert('RGB')

    boxes = detect(
        source,
        text_threshold = text_threshold,
        link_threshold = link_threshold,
        low_text       = low_text,
        long_size      = long_size,
        cuda           = cuda,
    )

    if not boxes:
        return []

    w, h = source.size
    results = []
    for (x1, y1, x2, y2) in boxes:
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = source.crop((cx1, cy1, cx2, cy2))
        rec  = recognize_crop(crop)
        if rec.text.strip():
            results.append(RegionResult(
                text       = rec.text,
                confidence = rec.confidence,
                script     = rec.script,
                method     = rec.method,
                bbox       = (x1, y1, x2, y2),
            ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _draw_results(image: Image.Image, results: list[RegionResult]) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    out  = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    except Exception:
        font = ImageFont.load_default()

    script_colours = {
        'latin':   (80, 200,  80),
        'cjk':     (80, 140, 220),
        'hangul':  (220, 80, 140),
        'unknown': (180, 180, 180),
    }
    for r in results:
        colour = script_colours.get(r.script, (200, 200, 200))
        x1, y1, x2, y2 = r.bbox
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        label = f'{r.text[:20]}  [{r.script}]'
        draw.text((x1, max(0, y1 - 14)), label, font=font, fill=colour)

    return out


def main():
    import io
    # Windows consoles default to cp1252; switch stdout to UTF-8 so CJK
    # characters in recognized text don't crash the print statements.
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    p = argparse.ArgumentParser(description='CRAFT + routed OCR pipeline')
    p.add_argument('--image',           required=True,  help='Input image path')
    p.add_argument('--text-threshold',  type=float, default=0.7)
    p.add_argument('--link-threshold',  type=float, default=0.4)
    p.add_argument('--low-text',        type=float, default=0.4)
    p.add_argument('--long-size',       type=int,   default=1280)
    p.add_argument('--pad',             type=int,   default=4)
    p.add_argument('--show-boxes',      action='store_true',
                   help='Save annotated image with bounding boxes')
    p.add_argument('--save-crops',      action='store_true',
                   help='Save each detected crop as a numbered PNG')
    args = p.parse_args()

    print(f'[ocr_pipeline] Processing: {args.image}')
    source = Image.open(args.image).convert('RGB')

    results = recognize(
        source,
        text_threshold = args.text_threshold,
        link_threshold = args.link_threshold,
        low_text       = args.low_text,
        long_size      = args.long_size,
        pad            = args.pad,
    )

    print(f'\n[ocr_pipeline] {len(results)} region(s) recognized\n')
    print(f'{"#":>4}  {"Script":8}  {"Method":12}  {"Conf":6}  {"Text"}')
    print('-' * 70)
    for i, r in enumerate(results, 1):
        print(f'{i:>4}  {r.script:8}  {r.method:12}  {r.confidence:.2f}  {r.text[:50]}')

    if args.show_boxes:
        here     = Path(args.image)
        out_path = here.with_name(here.stem + '_ocr_pipeline.png')
        annotated = _draw_results(source, results)
        annotated.save(out_path)
        print(f'\n[ocr_pipeline] Annotated image saved: {out_path}')

    if args.save_crops:
        crops_dir = Path(args.image).parent / 'ocr_pipeline_crops'
        crops_dir.mkdir(exist_ok=True)
        w, h = source.size
        for i, r in enumerate(results, 1):
            x1, y1, x2, y2 = r.bbox
            crop = source.crop((x1, y1, x2, y2))
            crop.save(crops_dir / f'region_{i:03d}_{r.script}.png')
        print(f'[ocr_pipeline] Crops saved to: {crops_dir}/')


if __name__ == '__main__':
    main()
