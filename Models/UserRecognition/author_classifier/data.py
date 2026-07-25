"""Dataset and dataloader construction for authorship attribution.

Reads the splits written by Models/Datasets/build_chat.py.  Unlike the image
pipeline, splitting happens at dataset-build time (chronologically, per author),
so this module never re-splits — it only loads what build_chat.py produced.
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_SPLITS = ('train', 'val', 'test')


# -- text augmentation --------------------------------------------------------
# Deliberately conservative: authorship signal lives in function words,
# punctuation and casing, so any augment that normalises those destroys the
# label.  Token dropout is safe (it thins content without changing style);
# synonym replacement and back-translation are not, and are omitted.

class TokenDropout:
    """Randomly drop whole tokens. Thins topical content, preserves style."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, text: str) -> str:
        toks = text.split()
        if len(toks) < 4:
            return text
        kept = [t for t in toks if random.random() > self.p]
        return ' '.join(kept) if kept else text


class SpanDropout:
    """Drop one contiguous span. Simulates a partially-observed message."""

    def __init__(self, max_frac: float = 0.2):
        self.max_frac = max_frac

    def __call__(self, text: str) -> str:
        toks = text.split()
        if len(toks) < 8:
            return text
        span = max(1, int(len(toks) * random.uniform(0.05, self.max_frac)))
        start = random.randint(0, len(toks) - span)
        return ' '.join(toks[:start] + toks[start + span:])


def get_augment(level: str = 'light'):
    """Return a callable str -> str for the requested augmentation level."""
    if level == 'none':
        return lambda t: t
    if level == 'light':
        drop = TokenDropout(p=0.05)
        return lambda t: drop(t)
    if level == 'heavy':
        drop, span = TokenDropout(p=0.12), SpanDropout(max_frac=0.25)
        def _aug(t: str) -> str:
            if random.random() < 0.5:
                t = drop(t)
            if random.random() < 0.3:
                t = span(t)
            return t
        return _aug
    raise ValueError(f'Unknown augment {level!r}. Choose: none | light | heavy')


# -- dataset ------------------------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self, rows: list, tokenizer, max_length: int = 256, augment=None):
        self.rows       = rows
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.augment    = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row  = self.rows[idx]
        text = row['text']
        if self.augment:
            text = self.augment(text)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item, torch.tensor(row['label'], dtype=torch.long)


def load_split(dataset_dir, split: str) -> list:
    p = Path(dataset_dir) / f'{split}.jsonl'
    if not p.exists():
        raise FileNotFoundError(
            f'{p} not found. Build it first:\n'
            f'  python Models/Datasets/build_chat.py --guild <GUILD_ID>'
        )
    rows = []
    with p.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_label_map(dataset_dir) -> list:
    """Return class_names indexed by label int."""
    p = Path(dataset_dir) / 'label_map.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found. Run build_chat.py first.')
    raw = json.loads(p.read_text(encoding='utf-8'))
    by_label = {v['label']: v['username'] for v in raw.values()}
    return [by_label[i] for i in sorted(by_label)]


def get_dataloaders(
    dataset_dir,
    tokenizer,
    batch_size:       int = 16,
    max_length:       int = 256,
    augment:          str = 'light',
    num_workers:      int = 0,
    weighted_sampler: bool = True,
):
    """
    Build train/val/test loaders from a chat-dataset guild directory.

    Returns (train_loader, val_loader, test_loader, class_names).
    """
    dataset_dir = Path(dataset_dir)
    class_names = load_label_map(dataset_dir)
    rows = {s: load_split(dataset_dir, s) for s in _SPLITS}

    train_ds = ChatDataset(rows['train'], tokenizer, max_length, augment=get_augment(augment))
    val_ds   = ChatDataset(rows['val'],   tokenizer, max_length, augment=None)
    test_ds  = ChatDataset(rows['test'],  tokenizer, max_length, augment=None)

    if weighted_sampler and rows['train']:
        counts  = Counter(r['label'] for r in rows['train'])
        weights = [1.0 / counts[r['label']] for r in rows['train']]
        sampler = WeightedRandomSampler(weights, num_samples=len(rows['train']), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sampler_tag = 'weighted' if weighted_sampler else 'shuffle'
    print(f'[data] {dataset_dir.name}  |  {len(class_names)} authors  |  '
          f'{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  |  '
          f'{sampler_tag} sampler  |  max_len={max_length}')
    return train_loader, val_loader, test_loader, class_names
