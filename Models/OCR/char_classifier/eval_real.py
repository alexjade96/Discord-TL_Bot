"""Staged comparison on real screenshots, so a mismatch between training and use
shows up at the stage that causes it.

Stages (run from OCR/):
    vocab         ground-truth characters that are not classes at all
    boxes         classifier on hand-placed glyph boxes (model alone)
    segmentation  segment.py piece count against the true character count
    pipeline      CRAFT + routing + segmentation + classifier (end to end)
    easyocr       the bot's production OCR on the same images (baseline)

    python -m char_classifier.eval_real
    python -m char_classifier.eval_real --stages vocab boxes --checkpoints checkpoints

The annotation file holds real user screenshots' text, so it lives in a
gitignored data directory.  Format:

    {"images": [{"file": "shot.png",                   # relative to the JSON
                 "lines": ["full text, line by line"],
                 "runs": [{"script": "cjk",            # cjk | kana | hangul
                           "text": "...",
                           "box": [x0, y0, x1, y1]}]}]}

A run is a tight box around equal-width glyphs; Latin runs are not supported.
The pipeline stage always uses checkpoints/ (compare.py's own loader).
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from .data          import get_transforms
    from .labels        import label_to_char
    from .model_builder import create_model
    from .segment       import split_into_chars
    from .utils         import get_device
else:
    from data          import get_transforms
    from labels        import label_to_char
    from model_builder import create_model
    from segment       import split_into_chars
    from utils         import get_device

_OCR_DIR = Path(__file__).parent.parent
_REPO = _OCR_DIR.parent.parent
DEFAULT_SET = _REPO / 'Translation' / '0-Data' / 'Image' / 'data' / 'eval_set.json'
DEFAULT_CKPTS = _OCR_DIR / 'checkpoints'
STAGES = ('vocab', 'boxes', 'segmentation', 'pipeline', 'easyocr')

# Square crop side as a multiple of glyph width: the midpoint of
# render_chars_context.MIN_CROP_SCALE / MAX_CROP_SCALE (a test keeps them in step).
GLYPH_SCALE = 1.75
TILE_SIZE = 128

_SCRIPT_RANGES = {
    'latin':  [(0x21, 0x7E)],
    'kana':   [(0x3041, 0x30FF)],
    'hangul': [(0xAC00, 0xD7A3)],
    'cjk':    [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)],
}


def load_set(path) -> list:
    """Images from an annotation file, with 'path' resolved next to the file."""
    path = Path(path)
    images = json.loads(path.read_text(encoding='utf-8'))['images']
    for entry in images:
        entry['path'] = path.parent / entry['file']
    return images


def glyph_crops(image: Image.Image, run: dict) -> list:
    """[(char, tile)] for a run, cut the way the generator cuts training tiles:
    a square centred on the glyph, padded with background (never black)."""
    x0, y0, x1, y1 = run['box']
    text = run['text']
    width = (x1 - x0) / len(text)
    half = GLYPH_SCALE * width / 2
    region = image.crop((int(x0), int(y0), int(x1), int(y1)))
    pixels = np.asarray(region.convert('RGB')).reshape(-1, 3)
    background = tuple(int(v) for v in np.median(pixels, axis=0))
    out = []
    for i, ch in enumerate(text):
        cx, cy = x0 + (i + 0.5) * width, (y0 + y1) / 2
        box = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
        tile = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), background)
        inside = image.crop((max(box[0], 0), max(box[1], 0),
                             min(box[2], image.width), min(box[3], image.height)))
        tile.paste(inside, (max(-box[0], 0), max(-box[1], 0)))
        out.append((ch, tile.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)))
    return out


@torch.inference_mode()
def score_boxes(model, class_names: list, scripts, images: list, device) -> dict:
    """Classifier-only accuracy on the annotated runs of the given scripts."""
    eval_tf = get_transforms('none', grid_mode='none')[1]
    chars = [label_to_char(name) for name in class_names]
    known = set(chars)
    model.eval()
    total = in_vocab = correct = 0
    misses = []
    for entry in images:
        runs = [r for r in entry.get('runs', []) if r['script'] in scripts]
        if not runs:
            continue
        image = Image.open(entry['path']).convert('RGB')
        for run in runs:
            for ch, tile in glyph_crops(image, run):
                total += 1
                if ch not in known:
                    continue
                in_vocab += 1
                pred = chars[int(model(eval_tf(tile).unsqueeze(0).to(device)).argmax(1))]
                if pred == ch:
                    correct += 1
                else:
                    misses.append({'file': entry['file'], 'true': ch, 'pred': pred})
    return {'chars': total, 'in_vocab': in_vocab, 'correct': correct, 'misses': misses}


def matched_chars(truth: str, predicted: str, script: str) -> tuple:
    """(characters matched in order, ground-truth characters) for one script."""
    def only(text):
        return ''.join(c for c in text
                       if any(a <= ord(c) <= b for a, b in _SCRIPT_RANGES[script]))
    t, p = only(truth), only(predicted)
    blocks = difflib.SequenceMatcher(None, t, p, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks), len(t)


def _score_text(images: list, predictions: dict) -> dict:
    out = {}
    for script in _SCRIPT_RANGES:
        pairs = [matched_chars(' '.join(e['lines']), predictions[e['file']], script)
                 for e in images]
        total = sum(n for _, n in pairs)
        if total:
            out[script] = {'matched': sum(m for m, _ in pairs), 'chars': total}
    return out


def _reading_order(items: list) -> str:
    """Join (bbox, text) pairs top-to-bottom, then left-to-right."""
    lines: list = []
    for box, text in sorted(items, key=lambda t: t[0][1] + t[0][3]):
        centre = (box[1] + box[3]) / 2
        if lines and centre - lines[-1][0] < 10:
            lines[-1][1].append((box, text))
        else:
            lines.append((centre, [(box, text)]))
    return ' '.join(t for _, line in lines for _, t in sorted(line, key=lambda i: i[0][0]))


def _load_model(ckpt_dir: Path, device):
    names = json.loads((ckpt_dir / 'class_names.json').read_text())
    config = ckpt_dir / 'config.json'
    backbone = 'dinov2_vits14'
    if config.exists():
        backbone = json.loads(config.read_text()).get('backbone', backbone)
    ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device, weights_only=False)
    if ckpt.get('class_names', names) != names:
        raise ValueError(f'{ckpt_dir.name}: best.pt and class_names.json list different classes')
    model = create_model(backbone, num_classes=len(names), freeze_base=False, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), names


def run(set_path, ckpt_root, stages) -> dict:
    images = load_set(set_path)
    ckpt_root = Path(ckpt_root)
    device = get_device()
    report: dict = {'eval_set': Path(set_path).name, 'images': len(images)}
    scripts = sorted({r['script'] for e in images for r in e.get('runs', [])})

    if 'vocab' in stages:
        report['vocab'] = {}
        for script in scripts:
            names_path = ckpt_root / script / 'class_names.json'
            if not names_path.exists():
                continue
            known = {label_to_char(n) for n in json.loads(names_path.read_text())}
            truth = ''.join(r['text'] for e in images for r in e.get('runs', [])
                            if r['script'] == script)
            report['vocab'][script] = {
                'chars': len(truth),
                'in_vocab': sum(c in known for c in truth),
                'missing': sorted(set(truth) - known),
            }

    if 'boxes' in stages:
        report['boxes'] = {}
        for script in scripts:
            if not (ckpt_root / script / 'best.pt').exists():
                continue
            model, names = _load_model(ckpt_root / script, device)
            report['boxes'][script] = score_boxes(model, names, [script], images, device)

    if 'segmentation' in stages:
        rows = []
        for entry in images:
            image = Image.open(entry['path']).convert('RGB')
            for r in entry.get('runs', []):
                pieces = len(split_into_chars(image.crop(tuple(int(v) for v in r['box']))))
                rows.append({'file': entry['file'], 'expected': len(r['text']), 'got': pieces})
        report['segmentation'] = {
            'exact_runs': sum(r['expected'] == r['got'] for r in rows),
            'runs': len(rows), 'detail': rows,
        }

    if 'pipeline' in stages:
        sys.path.insert(0, str(_OCR_DIR))
        import compare
        routed, classified = {}, {}
        for entry in images:
            regions = compare.compare_image(Image.open(entry['path']).convert('RGB'))
            routed[entry['file']] = _reading_order([(r.bbox, r.std_text) for r in regions])
            classified[entry['file']] = _reading_order([(r.bbox, r.clf_text) for r in regions])
        report['pipeline'] = {
            'router': _score_text(images, routed),
            'segment_plus_classifier': _score_text(images, classified),
        }

    if 'easyocr' in stages:
        sys.path[:0] = [str(_REPO / 'Translation' / '2-Image'),
                        str(_REPO / 'Translation' / '1-Text')]
        import ocr
        predictions = {e['file']: ' '.join(s['text'] for s in ocr.extract_text(e['path']))
                       for e in images}
        report['easyocr'] = _score_text(images, predictions)

    return report


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description='Staged OCR comparison on real screenshots')
    p.add_argument('--set', default=str(DEFAULT_SET), help='Annotation JSON')
    p.add_argument('--checkpoints', default=str(DEFAULT_CKPTS))
    p.add_argument('--stages', nargs='+', default=list(STAGES), choices=list(STAGES))
    p.add_argument('--out', default=None,
                   help='Report path (default: <checkpoints>/real_report.json)')
    args = p.parse_args()

    if not Path(args.set).exists():
        sys.exit(f'[eval_real] No annotation file at {args.set}')
    report = run(args.set, args.checkpoints, args.stages)

    for script, v in report.get('vocab', {}).items():
        print(f'[vocab]        {script}: {v["in_vocab"]}/{v["chars"]} ground-truth chars '
              f'are classes; missing {"".join(v["missing"]) or "none"}')
    for script, v in report.get('boxes', {}).items():
        print(f'[boxes]        {script}: {v["correct"]}/{v["in_vocab"]} in-vocab correct '
              f'({v["chars"]} annotated)')
    if 'segmentation' in report:
        s = report['segmentation']
        print(f'[segmentation] {s["exact_runs"]}/{s["runs"]} runs split into the right count')
    for name, scores in [*report.get('pipeline', {}).items(),
                         *([('easyocr', report['easyocr'])] if 'easyocr' in report else [])]:
        for script, v in scores.items():
            print(f'[{name}] {script}: {v["matched"]}/{v["chars"]} chars matched in order')

    out = Path(args.out) if args.out else Path(args.checkpoints) / 'real_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[eval_real] Report saved to {out}')


if __name__ == '__main__':
    main()
