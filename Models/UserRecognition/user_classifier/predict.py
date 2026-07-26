"""Single-text inference for a trained user recognition classifier.

Loads from a checkpoint directory (checkpoints/<guild_id>/) rather than the
deployed model root, so a model can be inspected before deploy.py installs it.

CLI:
    python -m user_classifier.predict --guild <ID> --text "some message"
    python -m user_classifier.predict --guild <ID> --text "..." --top 3
"""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .model_builder import create_model, create_tokenizer
    from .utils         import get_device
else:
    from model_builder import create_model, create_tokenizer
    from utils         import get_device


_HERE          = Path(__file__).parent
_DEFAULT_CKPTS = _HERE.parent / 'checkpoints'

# {checkpoint_dir: (model, tokenizer, class_names, max_length, device)}
_cache: dict = {}


def load(checkpoint_dir) -> tuple:
    """Load model + tokenizer from a checkpoint dir. Cached at module level."""
    key = str(checkpoint_dir)
    if key in _cache:
        return _cache[key]

    d = Path(checkpoint_dir)
    ckpt_path   = d / 'best.pt'
    config_path = d / 'config.json'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'No best.pt in {d}. Train first:\n'
            f'  python -m user_classifier.train --guild <GUILD_ID>'
        )
    if not config_path.exists():
        raise FileNotFoundError(f'No config.json in {d} — cannot determine the backbone.')

    config = json.loads(config_path.read_text(encoding='utf-8'))
    device = get_device()
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt.get('class_names', [])

    model = create_model(config['backbone'], num_classes=len(class_names), freeze_base=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    tokenizer = create_tokenizer(config['backbone'])

    _cache[key] = (model, tokenizer, class_names, config.get('max_length', 256), device)
    return _cache[key]


def predict(text: str, checkpoint_dir) -> list:
    """Return [{'username', 'label', 'score'}, ...] sorted by score descending."""
    model, tokenizer, class_names, max_length, device = load(checkpoint_dir)
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    padding='max_length', return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**enc)[0], dim=-1).cpu().tolist()

    results = [
        {'username': class_names[i] if i < len(class_names) else str(i),
         'label': i,
         'score': round(float(p), 4)}
        for i, p in enumerate(probs)
    ]
    results.sort(key=lambda r: r['score'], reverse=True)
    return results


def main():
    p = argparse.ArgumentParser(description='Rank likely senders of a message.')
    p.add_argument('--guild', required=True, help='Guild ID')
    p.add_argument('--text',  required=True, help='Message text to classify')
    p.add_argument('--top',   type=int, default=0, help='Show top N results (default: all)')
    p.add_argument('--checkpoint-dir', default=None, help='Override checkpoints/<guild_id>/')
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir \
        else _DEFAULT_CKPTS / str(args.guild)

    results = predict(args.text, ckpt_dir)
    if args.top:
        results = results[: args.top]

    print(f'Text: {args.text!r}\n')
    for rank, r in enumerate(results, 1):
        filled = int(r['score'] * 20)
        bar = '#' * filled + '.' * (20 - filled)
        print(f"  {rank}. {r['username']:<20} {bar}  {r['score'] * 100:.1f}%")


if __name__ == '__main__':
    main()
