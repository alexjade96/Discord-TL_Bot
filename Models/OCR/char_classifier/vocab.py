"""Vocabulary checks: does a class list cover the characters people actually type?

Test accuracy only measures the classes a model has.  This compares the class
list against standard character sets, so a narrow or malformed list shows up
before data is rendered and before a training run, not after promotion.

Run from OCR/:
    python -m char_classifier.vocab                  # every checkpoints/<script>/
    python -m char_classifier.vocab --check-weights  # also compare .pt class lists

Console output is ASCII only (characters are shown as U+XXXX).
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

if __package__:
    from .labels import label_script, label_to_char
else:
    from labels import label_script, label_to_char

_CKPT_ROOT = Path(__file__).parent.parent / 'checkpoints'
SCRIPTS = ('latin', 'kana', 'hangul', 'cjk')

# Unicode ranges a class of each script is expected to fall in.
_BLOCKS = {
    'latin':  [(0x21, 0x7E)],
    'kana':   [(0x3041, 0x30FF)],
    'hangul': [(0xAC00, 0xD7A3)],
    'cjk':    [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)],
}

# Share of the achievable coverage below which a reference set is flagged.
DEFAULT_WARN_BELOW = 0.9


def _kuten_rows(codec: str, first: int, last: int) -> set:
    """Characters in rows first..last of a 94x94 East Asian code table."""
    out = set()
    for row in range(first, last + 1):
        for cell in range(1, 95):
            try:
                out.add(bytes([0xA0 + row, 0xA0 + cell]).decode(codec))
            except UnicodeDecodeError:
                pass
    return out


@lru_cache(maxsize=None)
def reference_sets(script: str) -> dict:
    """{name: frozenset of characters}, built from Python's own codecs."""
    if script == 'latin':
        sets = {'ASCII letters and digits': {chr(c) for c in range(128) if chr(c).isalnum()}}
    elif script == 'kana':
        sets = {
            'basic hiragana and katakana':
                {chr(c) for c in (*range(0x3041, 0x3094), *range(0x30A1, 0x30F7))},
            'long vowel mark': {'ー'},
        }
    elif script == 'hangul':
        sets = {'KS X 1001 syllables': _kuten_rows('euc_kr', 16, 40)}
    elif script == 'cjk':
        sets = {
            'GB2312 level 1 (simplified Chinese)': _kuten_rows('gb2312', 16, 55),
            'JIS X 0208 level 1 (kanji)': _kuten_rows('euc_jp', 16, 47),
        }
    else:
        sets = {}
    return {name: frozenset(chars) for name, chars in sets.items()}


def vocab_report(class_names: list, script: str) -> dict:
    """Coverage and sanity figures for one script's class labels."""
    decoded = [(label, label_to_char(label)) for label in class_names]
    chars = [c for _, c in decoded if len(c) == 1]
    blocks = _BLOCKS.get(script, [])
    have = set(chars)
    coverage = {}
    for name, ref in reference_sets(script).items():
        covered = len(have & ref)
        achievable = min(len(ref), len(class_names))
        coverage[name] = {
            'covered': covered,
            'total': len(ref),
            'fraction': covered / len(ref),
            # fraction of what a list of this size could have covered
            'of_achievable': covered / achievable if achievable else 0.0,
        }
    return {
        'n_classes': len(class_names),
        'undecodable': [label for label, c in decoded if len(c) != 1],
        'duplicate_chars': len(chars) - len(have),
        'out_of_script': sorted(
            label for label, c in decoded
            if len(c) == 1 and blocks and not any(a <= ord(c) <= b for a, b in blocks)),
        'coverage': coverage,
    }


def vocab_reports(class_names: list) -> dict:
    """{script: report} for a class list that may mix several scripts."""
    grouped: dict = {}
    for label in class_names:
        grouped.setdefault(label_script(label), []).append(label)
    return {script: vocab_report(labels, script) for script, labels in grouped.items()}


def failures(reports: dict, min_of_achievable: float) -> list:
    """Human-readable problems in a vocab_reports() result."""
    out = []
    for script, rep in reports.items():
        if script == 'unknown':
            out.append(f'{rep["n_classes"]} label(s) with an unknown prefix')
        if rep['undecodable']:
            out.append(f'{script}: {len(rep["undecodable"])} undecodable label(s), '
                       f'e.g. {ascii(rep["undecodable"][:3])}')
        if rep['duplicate_chars']:
            out.append(f'{script}: {rep["duplicate_chars"]} duplicate character(s)')
        if rep['out_of_script']:
            out.append(f'{script}: {len(rep["out_of_script"])} class(es) outside the script, '
                       f'e.g. {ascii(rep["out_of_script"][:3])}')
        for name, cov in rep['coverage'].items():
            if cov['of_achievable'] < min_of_achievable:
                out.append(f'{script}: {name} coverage {cov["covered"]}/{cov["total"]} '
                           f'({cov["of_achievable"]:.0%} of achievable, '
                           f'floor {min_of_achievable:.0%})')
    return out


def format_reports(reports: dict, warn_below: float = DEFAULT_WARN_BELOW,
                   tag: str = 'vocab') -> list:
    lines = []
    for script, rep in reports.items():
        for name, cov in rep['coverage'].items():
            lines.append(f'[{tag}] {script}: {rep["n_classes"]} classes, {name}: '
                         f'{cov["covered"]}/{cov["total"]} ({cov["fraction"]:.1%}, '
                         f'{cov["of_achievable"]:.0%} of achievable)')
    lines += [f'[{tag}] WARNING {msg}' for msg in failures(reports, warn_below)]
    return lines


def check_weights(ckpt_dir: Path, class_names: list) -> list:
    """Problems where a .pt file's class list differs from class_names.json."""
    import torch
    out = []
    for name in ('best.pt', 'last.pt'):
        path = ckpt_dir / name
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        stored = ckpt.get('class_names')
        if stored != class_names:
            out.append(f'{ckpt_dir.name}/{name}: {len(stored or [])} classes in the file, '
                       f'{len(class_names)} in class_names.json (lists differ)')
    return out


def main():
    p = argparse.ArgumentParser(description='Check class vocabularies against standard sets')
    p.add_argument('--checkpoints', default=str(_CKPT_ROOT))
    p.add_argument('--scripts', nargs='+', default=list(SCRIPTS), choices=list(SCRIPTS))
    p.add_argument('--check-weights', action='store_true',
                   help='Also verify best.pt/last.pt carry the same class list (loads the files)')
    p.add_argument('--min-coverage', type=float, default=DEFAULT_WARN_BELOW,
                   help='Exit 1 if any set is below this share of achievable coverage')
    args = p.parse_args()

    problems = []
    for script in args.scripts:
        ckpt_dir = Path(args.checkpoints) / script
        names_path = ckpt_dir / 'class_names.json'
        if not names_path.exists():
            print(f'[vocab] {script}: no class_names.json in {ckpt_dir.name}/, skipped')
            continue
        names = json.loads(names_path.read_text())
        reports = vocab_reports(names)
        for line in format_reports(reports, args.min_coverage):
            print(line)
        problems += failures(reports, args.min_coverage)
        if args.check_weights:
            mismatched = check_weights(ckpt_dir, names)
            for msg in mismatched:
                print(f'[vocab] WARNING {msg}')
            problems += mismatched
    print(f'[vocab] {len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
