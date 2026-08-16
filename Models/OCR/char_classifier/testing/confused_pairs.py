"""Confused-pairs diagnostic for a trained char_classifier checkpoint.

Rebuilds the test split a checkpoint was trained against (same seed /
min_per_class / max_per_class recorded in its config.json), runs inference
on a random subsample, and reports the top confused pairs -- flagging
rotation-ambiguous pairs and the low_i collapse specifically (see data.py's
crop-scale comment for the mechanism this is checking).

Run from Models/OCR/:
    python -m char_classifier.testing.confused_pairs --checkpoint checkpoints/latin/best.pt

Or against an external checkpoint (e.g. the Drive-mounted Colab run):
    python -m char_classifier.testing.confused_pairs \
        --checkpoint "G:/My Drive/Colab Notebooks/TL-Bot/checkpoints/latin/best.pt"
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__:
    from ..data          import build_dataset, get_transforms, CharDataset
    from ..model_builder import create_model
    from ..utils         import get_device, load_checkpoint
else:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data          import build_dataset, get_transforms, CharDataset
    from model_builder import create_model
    from utils         import get_device, load_checkpoint


_HERE          = Path(__file__).parent
_DATASET_ROOT  = _HERE.parent.parent.parent / 'Datasets' / 'char-dataset'

# Known rotation-ambiguous Latin pairs -- glyphs that become each other under
# rotation up to 270deg (see colab_train.ipynb's GRID_MODE note). Flagged
# separately from the low_i collapse since they're a distinct, expected
# source of confusion rather than the crop-fragment mechanism under test.
_ROTATION_PAIRS = {
    frozenset({'low_b', 'low_q'}), frozenset({'low_d', 'low_p'}),
    frozenset({'low_n', 'low_u'}), frozenset({'dig_6', 'dig_9'}),
    frozenset({'cap_M', 'cap_W'}),
}


def parse_args():
    p = argparse.ArgumentParser(description='Confused-pairs diagnostic for char_classifier')
    p.add_argument('--checkpoint', required=True, help='Path to a best.pt / last.pt checkpoint')
    p.add_argument('--config', default=None,
                   help='config.json to read seed/min_per_class/max_per_class/scripts from '
                        '(default: config.json next to --checkpoint)')
    p.add_argument('--dataset-dir', nargs='+', default=None,
                   help='Override dataset dir(s) (default: char-dataset/<script> from config)')
    p.add_argument('--scripts', nargs='+', default=None,
                   help='Override script(s) if config.json is missing/incomplete')
    p.add_argument('--backbone', default=None, help='Override backbone (default: from config)')
    p.add_argument('--sample-size', type=int, default=2500,
                   help='Random subsample of the test split to evaluate (default 2500)')
    p.add_argument('--top-k', type=int, default=25, help='Number of confused pairs to report')
    p.add_argument('--seed', type=int, default=42, help='Fallback seed if config.json lacks one')
    p.add_argument('--out', default=None, help='Optional path to write a JSON report')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path   = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else ckpt_path.parent / 'config.json'
    config      = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not config_path.exists():
        print(f'[confused_pairs] WARNING: no config.json at {config_path} -- '
              f'falling back to CLI defaults/args, split may not match training exactly.')

    scripts       = args.scripts or config.get('scripts', ['latin'])
    backbone      = args.backbone or config.get('backbone', 'dinov2_vits14')
    seed          = config.get('seed', args.seed)
    min_per_class = config.get('min_per_class', 5)
    max_per_class = config.get('max_per_class', 0)
    # 'dataset_name' only exists in config.json written after the
    # render_chars_context.py addition; older checkpoints (e.g. run 4) predate
    # it and fall back to the original char-dataset, which is what they were
    # actually trained on.
    dataset_name  = config.get('dataset_name', 'char-dataset')
    dataset_dirs  = args.dataset_dir or [str(_DATASET_ROOT.parent / dataset_name / s) for s in scripts]

    print(f'[confused_pairs] checkpoint = {ckpt_path}')
    print(f'[confused_pairs] scripts={scripts}  backbone={backbone}  seed={seed}  '
          f'dataset={dataset_name}  min_per_class={min_per_class}  max_per_class={max_per_class}')

    device = get_device()
    print(f'[confused_pairs] device = {device}')

    _, _, test_s, class_names, _ = build_dataset(
        dataset_dirs, seed=seed, max_per_class=max_per_class, min_per_class=min_per_class,
    )
    print(f'[confused_pairs] test split: {len(test_s)} images, {len(class_names)} classes')

    rng    = random.Random(seed)
    sample = test_s if len(test_s) <= args.sample_size else rng.sample(test_s, args.sample_size)
    print(f'[confused_pairs] evaluating on {len(sample)} sampled images')

    _, eval_tf = get_transforms('none')
    loader = DataLoader(CharDataset(sample, transform=eval_tf), batch_size=64,
                        shuffle=False, num_workers=0)

    model = create_model(backbone, num_classes=len(class_names), freeze_base=True).to(device)
    epoch, val_acc, ckpt_class_names = load_checkpoint(str(ckpt_path), model, device=device)
    if ckpt_class_names and ckpt_class_names != class_names:
        print('[confused_pairs] WARNING: checkpoint class_names differ from the rebuilt '
              'dataset class_names -- predictions may be misaligned to the wrong labels.')
    model.eval()
    print(f'[confused_pairs] loaded checkpoint from epoch {epoch}  (recorded val_acc {val_acc:.4f})')

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X, y in loader:
            logits = model(X.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    pair_counts = Counter()
    for pred, label in zip(all_preds, all_labels):
        if pred != label:
            pair_counts[(class_names[label], class_names[pred])] += 1

    total_errors = sum(pair_counts.values())
    top1 = 1.0 - total_errors / len(all_labels)
    print(f'\n[confused_pairs] sample top-1 acc: {top1:.4f}  '
          f'({total_errors} errors / {len(all_labels)} samples)')

    low_i_errors = sum(c for (t, p), c in pair_counts.items() if t == 'low_i' or p == 'low_i')
    rotation_errors = sum(
        c for (t, p), c in pair_counts.items() if frozenset({t, p}) in _ROTATION_PAIRS
    )
    low_i_frac    = low_i_errors / max(total_errors, 1)
    rotation_frac = rotation_errors / max(total_errors, 1)
    print(f'[confused_pairs] errors involving low_i (as true or pred): '
          f'{low_i_errors} ({100*low_i_frac:.1f}% of errors)')
    print(f'[confused_pairs] errors matching known rotation-ambiguous pairs: '
          f'{rotation_errors} ({100*rotation_frac:.1f}% of errors)')

    ranked = pair_counts.most_common(args.top_k)
    print(f'\n--- Top {args.top_k} Confused Pairs (true -> predicted) ---')
    rows = []
    for (true_cls, pred_cls), count in ranked:
        flags = []
        if true_cls == 'low_i' or pred_cls == 'low_i':
            flags.append('low_i')
        if frozenset({true_cls, pred_cls}) in _ROTATION_PAIRS:
            flags.append('rotation')
        flag_str = f"  [{', '.join(flags)}]" if flags else ''
        print(f'  {true_cls:20s} -> {pred_cls:20s}  ({count}){flag_str}')
        rows.append({'true': true_cls, 'pred': pred_cls, 'count': int(count), 'flags': flags})

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'checkpoint':             str(ckpt_path),
            'checkpoint_epoch':       epoch,
            'checkpoint_val_acc':     val_acc,
            'scripts':                scripts,
            'sample_size':            len(sample),
            'sample_top1_acc':        top1,
            'total_errors':           total_errors,
            'low_i_error_count':      low_i_errors,
            'low_i_error_fraction':   low_i_frac,
            'rotation_error_count':   rotation_errors,
            'rotation_error_fraction': rotation_frac,
            'top_confused_pairs':     rows,
        }, indent=2))
        print(f'\n[confused_pairs] saved report to {out_path}')


if __name__ == '__main__':
    main()
