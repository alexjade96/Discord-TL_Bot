import io
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from grid_augments import (  # noqa: F401  (re-exported for callers)
    TileGrid3x3, TileGrid3x3Rotated,
    TileGrid3x3Pair, TileGrid3x3PairRotated,
    TileGrid3x3Orbital, TileGrid3x3OrbitalRotated,
    RandomGridAugment,
    _GRID_ANGLES,
)


# -- custom transforms --------------------------------------------------------

class AddGaussianNoise:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t + torch.randn_like(t) * self.std).clamp(0.0, 1.0)


class SimulateJPEG:
    def __init__(self, quality_range=(30, 85)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


# -- transforms ---------------------------------------------------------------
# Source images are single centered character tiles (TILE_SIZE×TILE_SIZE).
# TileGrid3x3 assembles a 3×3 grid before RandomResizedCrop so the crop can
# land on the centre character, a neighbour, or a boundary between two —
# simulating how a character looks inside a word or line of text.
# Horizontal flip is intentionally omitted: it would mirror b↔d, p↔q, etc.
# Crop scale (0.20–0.45) on the 3× wider grid gives ~0.6–1.4 char widths.

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# ---------------------------------------------------------------------------
# Grid mode helpers
# ---------------------------------------------------------------------------
# 'single'  — TileGrid3x3 only (original default; fastest)
# 'rotated' — TileGrid3x3 + TileGrid3x3Rotated (adds skew/angle invariance)
# 'all'     — all six variants chosen at random per sample
_GRID_MODES = ('single', 'rotated', 'all')


def _build_grid_tf(mode: str, peer_pool: list | None):
    """Return the appropriate grid augmentation transform for the given mode."""
    if mode == 'single':
        return TileGrid3x3(peer_pool=peer_pool)
    if mode == 'rotated':
        return RandomGridAugment([
            TileGrid3x3(peer_pool=peer_pool),
            TileGrid3x3Rotated(peer_pool=peer_pool),
        ])
    if mode == 'all':
        return RandomGridAugment([
            TileGrid3x3(peer_pool=peer_pool),
            TileGrid3x3Rotated(peer_pool=peer_pool),
            TileGrid3x3Pair(peer_pool=peer_pool),
            TileGrid3x3PairRotated(peer_pool=peer_pool),
            TileGrid3x3Orbital(),
            TileGrid3x3OrbitalRotated(),
        ])
    raise ValueError(f'Unknown grid_mode {mode!r}. Choose: {" | ".join(_GRID_MODES)}')


def get_transforms(augment: str = 'heavy', peer_pool: list | None = None,
                   grid_mode: str = 'single'):
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _NORMALIZE,
    ])

    if augment == 'none':
        return eval_tf, eval_tf

    base = [
        transforms.RandomResizedCrop(
            224,
            scale=(0.75, 1.0) if augment == 'light' else (0.20, 0.45),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1),
    ]

    if augment == 'heavy':
        train_tf = transforms.Compose([
            SimulateJPEG(quality_range=(40, 90)),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            _build_grid_tf(grid_mode, peer_pool),
            *base,
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            AddGaussianNoise(std=0.04),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            _NORMALIZE,
        ])
    else:
        train_tf = transforms.Compose([*base, transforms.ToTensor(), _NORMALIZE])

    return train_tf, eval_tf


# -- dataset ------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataset(
    dataset_dirs,
    val_split:     float = 0.15,
    test_split:    float = 0.15,
    seed:          int   = 42,
    max_per_class: int   = 0,    # 0 = no cap; >0 = cap each class at N images
    min_per_class: int   = 5,    # skip classes with fewer images than this
):
    """
    Build train/val/test splits from one or more class-directory trees.

    dataset_dirs : str | Path | list[str | Path]
        Each entry is a directory whose subdirectories are class folders
        containing PNG images.  Multiple directories are merged into a single
        flat class space — class names across directories must be unique
        (guaranteed when using render_chars output: latin uses 'A/'/'B'/...,
        kana uses 'hira_3041/'/'kata_30a2'/..., hangul 'syl_ac00/'...,
        cjk 'cjk_4e00/'...).
    """
    if isinstance(dataset_dirs, (str, Path)):
        roots = [Path(dataset_dirs)]
    else:
        roots = [Path(d) for d in dataset_dirs]

    rng = random.Random(seed)
    per_class: dict[str, list] = {}
    for root in roots:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            imgs = sorted(d.glob('*.png'))
            if len(imgs) >= min_per_class:
                if d.name in per_class:
                    print(f'[data] WARNING: duplicate class name {d.name!r} from {root} — skipping')
                else:
                    if max_per_class > 0 and len(imgs) > max_per_class:
                        imgs = rng.sample(imgs, max_per_class)
                    per_class[d.name] = imgs
            else:
                if len(imgs) > 0:
                    print(f'[data] skipping {d.name!r}: only {len(imgs)} image(s) (need {min_per_class})')

    class_names  = sorted(per_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_samples = [
        (p, class_to_idx[cls])
        for cls in class_names
        for p in per_class[cls]
    ]

    if not all_samples:
        raise ValueError(f'No training images found in {[str(r) for r in roots]}')

    paths, labels = zip(*all_samples)
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=val_split + test_split, stratify=labels, random_state=seed,
    )
    val_ratio = val_split / (val_split + test_split)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=1.0 - val_ratio, stratify=None, random_state=seed,
    )
    return (
        list(zip(train_p, train_l)),
        list(zip(val_p,   val_l)),
        list(zip(test_p,  test_l)),
        class_names,
        class_to_idx,
    )


def get_dataloaders(
    dataset_dirs,
    batch_size:       int   = 32,
    augment:          str   = 'heavy',
    grid_mode:        str   = 'single',
    val_split:        float = 0.15,
    test_split:       float = 0.15,
    seed:             int   = 42,
    num_workers:      int   = 0,
    weighted_sampler: bool  = True,
    max_per_class:    int   = 0,
    min_per_class:    int   = 5,
):
    """
    dataset_dirs : str | Path | list[str | Path]
        Forwarded to build_dataset.  Pass a list to train on multiple scripts.
    max_per_class : int
        Cap each class at N images before splitting (0 = no cap).
        Useful for fast smoke tests without changing dataset structure.
    min_per_class : int
        Minimum images required to include a class (default 5).
        Lower to 1 if you want to include classes rendered by only one font.
    """
    train_s, val_s, test_s, class_names, _ = build_dataset(
        dataset_dirs, val_split, test_split, seed,
        max_per_class=max_per_class, min_per_class=min_per_class,
    )
    train_tf, eval_tf = get_transforms(augment, grid_mode=grid_mode)
    train_ds = CharDataset(train_s, transform=train_tf)
    val_ds   = CharDataset(val_s,   transform=eval_tf)
    test_ds  = CharDataset(test_s,  transform=eval_tf)

    if weighted_sampler:
        counts  = Counter(label for _, label in train_s)
        weights = [1.0 / counts[label] for _, label in train_s]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_s), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    sampler_tag = 'weighted' if weighted_sampler else 'shuffle'
    scripts_tag = ', '.join(
        Path(d).name if isinstance(d, (str, Path)) else str(d)
        for d in ([dataset_dirs] if isinstance(dataset_dirs, (str, Path)) else dataset_dirs)
    )
    print(f'[data] scripts={scripts_tag}  |  {len(class_names)} classes  |  '
          f'{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  |  '
          f'{sampler_tag} sampler')
    return train_loader, val_loader, test_loader, class_names
