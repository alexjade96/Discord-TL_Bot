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


# -- custom transforms --------------------------------------------------------

class AddGaussianNoise:
    # Add per-pixel Gaussian noise; applied after ToTensor
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t + torch.randn_like(t) * self.std).clamp(0.0, 1.0)


class SimulateJPEG:
    # Round-trip through JPEG encoder to introduce compression artifacts
    def __init__(self, quality_range=(30, 85)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomBackground:
    # Randomly invert image to simulate light-text-on-dark-background
    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return Image.fromarray(255 - np.array(img))
        return img


# -- transforms ---------------------------------------------------------------
# Source images are 2048x1024 glyph-grid pages.
# RandomResizedCrop with small scale crops individual glyph clusters,
# matching real-world usage (identifying a font from a text excerpt).

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_transforms(augment: str = 'heavy'):
    # Returns (train_transform, eval_transform). augment: 'none' | 'light' | 'heavy'
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
            scale=(0.1, 0.5) if augment == 'light' else (0.05, 0.35),
            ratio=(0.75, 1.33),
        ),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    ]

    if augment == 'heavy':
        train_tf = transforms.Compose([
            SimulateJPEG(quality_range=(40, 90)),
            RandomBackground(p=0.25),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.35),
            *base,
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            AddGaussianNoise(std=0.04),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            _NORMALIZE,
        ])
    else:
        train_tf = transforms.Compose([*base, transforms.ToTensor(), _NORMALIZE])

    return train_tf, eval_tf


# -- dataset ------------------------------------------------------------------

class FontDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataset(dataset_dir, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
    # Scan dataset_dir for font-family folders and split into train/val/test.
    # Excludes *_preview.png (full glyph-sheet composites).
    # Classes with fewer than 3 images are skipped (need ≥1 sample in each split).
    root = Path(dataset_dir)
    all_class_names = sorted(d.name for d in root.iterdir() if d.is_dir())

    per_class = {}
    for cls in all_class_names:
        imgs = [p for p in sorted((root / cls).glob('*.png')) if '_preview' not in p.name]
        if len(imgs) >= 5:
            per_class[cls] = imgs
        else:
            print(f'[data] skipping {cls!r}: only {len(imgs)} image(s)')

    class_names = sorted(per_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_samples = [
        (p, class_to_idx[cls])
        for cls in class_names
        for p in per_class[cls]
    ]

    if not all_samples:
        raise ValueError(f'No training images found in {root}')

    paths, labels = zip(*all_samples)
    # Stratify the first split only — ensures every class is in train.
    # Second split (val vs test) is random; stratification there would need
    # ≥2 samples per class in temp, which many 3-image families can't satisfy.
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
    dataset_dir,
    batch_size: int = 32,
    augment: str = 'heavy',
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    weighted_sampler: bool = True,
):
    train_s, val_s, test_s, class_names, _ = build_dataset(
        dataset_dir, val_split, test_split, seed
    )
    train_tf, eval_tf = get_transforms(augment)
    train_ds = FontDataset(train_s, transform=train_tf)
    val_ds   = FontDataset(val_s,   transform=eval_tf)
    test_ds  = FontDataset(test_s,  transform=eval_tf)

    if weighted_sampler:
        counts  = Counter(label for _, label in train_s)
        weights = [1.0 / counts[label] for _, label in train_s]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_s), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sampler_tag = 'weighted' if weighted_sampler else 'shuffle'
    print(f'[data] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  |  {len(class_names)} classes  |  {sampler_tag} sampler')
    return train_loader, val_loader, test_loader, class_names
