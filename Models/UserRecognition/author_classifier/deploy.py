"""Install a trained authorship classifier for local inference.

Copies checkpoints/<guild_id>/ into ~/.tl-bot/authorship/<guild_id>/ so
UserRecognition/identify.py loads it automatically. identify.py dispatches on
the "model_type" field this script writes into meta.json, so deploying here
takes over from the TF-IDF baseline without any code change in the bot.

Usage:
    python -m author_classifier.deploy --guild <ID>
    python -m author_classifier.deploy --guild <ID> --checkpoint path/to/last.pt
    python -m author_classifier.deploy --list
    python -m author_classifier.deploy --remove <ID>          # revert to TF-IDF
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

_HERE          = Path(__file__).parent
_DEFAULT_CKPTS = _HERE.parent / 'checkpoints'
_MODEL_ROOT    = Path.home() / '.tl-bot' / 'authorship'

# Written by the TF-IDF baseline (UserRecognition/0-Data/training/train.py).
_TFIDF_FILES = ('word_vec.pkl', 'char_vec.pkl', 'clf.pkl')
_NEURAL_FILES = ('model.pt', 'config.json', 'class_names.json')


def deploy(guild_id: str, checkpoint: Path | None = None,
           checkpoint_dir: Path | None = None, keep_tfidf: bool = True) -> dict:
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CKPTS / str(guild_id)
    src_ckpt = Path(checkpoint) if checkpoint else ckpt_dir / 'best.pt'
    config   = ckpt_dir / 'config.json'
    classes  = ckpt_dir / 'class_names.json'

    for p in (src_ckpt, config, classes):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not found. Train first:\n'
                f'  python -m author_classifier.train --guild {guild_id}'
            )

    dest = _MODEL_ROOT / str(guild_id)
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_ckpt, dest / 'model.pt')
    shutil.copy2(config,   dest / 'config.json')
    shutil.copy2(classes,  dest / 'class_names.json')

    # label_map.json is what identify.py reads for usernames; the TF-IDF path
    # already installs one, so only copy if the checkpoint dir carries its own.
    src_label_map = ckpt_dir / 'label_map.json'
    if src_label_map.exists():
        shutil.copy2(src_label_map, dest / 'label_map.json')

    cfg = json.loads(config.read_text(encoding='utf-8'))
    class_names = json.loads(classes.read_text(encoding='utf-8'))

    shadowed = [f for f in _TFIDF_FILES if (dest / f).exists()]
    if shadowed and not keep_tfidf:
        for f in shadowed:
            (dest / f).unlink()
        shadowed = []

    meta = {
        'model_type':  'neural',
        'guild_id':    str(guild_id),
        'backbone':    cfg.get('backbone'),
        'max_length':  cfg.get('max_length', 256),
        'num_classes': len(class_names),
        'source':      str(src_ckpt),
        'deployed_at': datetime.now().isoformat(timespec='seconds'),
    }
    (dest / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    return {'dest': str(dest), 'meta': meta, 'shadowed_tfidf': shadowed}


def list_installed() -> None:
    if not _MODEL_ROOT.exists():
        print(f'No models installed at {_MODEL_ROOT}.')
        return
    found = False
    for d in sorted(_MODEL_ROOT.iterdir()):
        if not d.is_dir():
            continue
        found = True
        has_neural = all((d / f).exists() for f in _NEURAL_FILES)
        has_tfidf  = all((d / f).exists() for f in _TFIDF_FILES)
        meta = {}
        if (d / 'meta.json').exists():
            try:
                meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                pass
        active = meta.get('model_type', 'tfidf' if has_tfidf else 'unknown')
        present = [n for n, ok in (('neural', has_neural), ('tfidf', has_tfidf)) if ok]
        print(f'  {d.name}  |  active: {active}  |  present: {", ".join(present) or "none"}')
        if active == 'neural':
            print(f'      backbone {meta.get("backbone")}  |  '
                  f'{meta.get("num_classes")} authors  |  '
                  f'deployed {meta.get("deployed_at", "?")}')
    if not found:
        print(f'No guild directories under {_MODEL_ROOT}.')


def remove(guild_id: str) -> dict:
    """Remove the neural model, reverting to TF-IDF if it is still installed."""
    dest = _MODEL_ROOT / str(guild_id)
    if not dest.exists():
        return {'removed': [], 'fallback': None}

    removed = []
    for f in (*_NEURAL_FILES, 'meta.json'):
        p = dest / f
        if p.exists():
            p.unlink()
            removed.append(f)

    fallback = 'tfidf' if all((dest / f).exists() for f in _TFIDF_FILES) else None
    if fallback is None and not any(dest.iterdir()):
        dest.rmdir()
    return {'removed': removed, 'fallback': fallback}


def main():
    p = argparse.ArgumentParser(description='Deploy an authorship classifier for local inference.')
    p.add_argument('--guild',      default=None, help='Guild ID to deploy')
    p.add_argument('--checkpoint', default=None, type=Path,
                   help='Checkpoint to install (default: checkpoints/<guild_id>/best.pt)')
    p.add_argument('--checkpoint-dir', default=None, type=Path,
                   help='Override checkpoints/<guild_id>/')
    p.add_argument('--replace-tfidf', action='store_true',
                   help='Delete the TF-IDF pickles instead of leaving them in place')
    p.add_argument('--list',   action='store_true', help='List installed models and exit')
    p.add_argument('--remove', default=None, metavar='GUILD_ID',
                   help='Uninstall a guild neural model (reverts to TF-IDF if present)')
    args = p.parse_args()

    if args.list:
        list_installed()
        return

    if args.remove:
        result = remove(args.remove)
        if not result['removed']:
            print(f'Nothing to remove for guild {args.remove}.')
        else:
            print(f'Removed {", ".join(result["removed"])} for guild {args.remove}.')
            print(f'Now serving: {result["fallback"] or "nothing — no model installed"}.')
        return

    if not args.guild:
        p.print_help()
        return

    result = deploy(args.guild, checkpoint=args.checkpoint,
                    checkpoint_dir=args.checkpoint_dir,
                    keep_tfidf=not args.replace_tfidf)
    print(f'Deployed authorship model for guild {args.guild}:')
    print(f'  Destination: {result["dest"]}')
    print(f'  Backbone:    {result["meta"]["backbone"]}')
    print(f'  Authors:     {result["meta"]["num_classes"]}')
    if result['shadowed_tfidf']:
        print(f'  Note: TF-IDF files still present ({", ".join(result["shadowed_tfidf"])}); '
              f'the neural model takes precedence. Use --replace-tfidf to delete them, '
              f'or --remove to revert.')


if __name__ == '__main__':
    main()
