import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(model, optimizer, epoch: int, val_acc: float, path, class_names: list,
                    meta: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
            'meta': meta or {},
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['val_acc'], ckpt.get('class_names', [])


def peek_checkpoint_epoch(path) -> int:
    """Return the saved epoch from a checkpoint without loading model weights."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt.get('epoch', 0)


def print_checkpoint_info(path):
    """Print a human-readable summary of a checkpoint's training state."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    epoch    = ckpt.get('epoch', '?')
    val_acc  = ckpt.get('val_acc', 0.0)
    meta     = ckpt.get('meta', {})
    print(f"  Checkpoint : {path}")
    print(f"  Epoch      : {epoch} / {meta.get('total_epochs', '?')}"
          f"  ({meta.get('epochs_remaining', '?')} remaining)")
    print(f"  Phase      : {meta.get('phase', '?')} — {meta.get('phase_label', '')}")
    print(f"  Val acc    : {val_acc:.4f}  (best so far: {meta.get('best_val_acc', val_acc):.4f})")
    print(f"  Scripts    : {meta.get('scripts', '?')}")
    print(f"  Backbone   : {meta.get('backbone', '?')}")
    print(f"  Saved at   : {meta.get('saved_at', 'unknown')}")
