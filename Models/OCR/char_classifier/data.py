import io
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


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


class TileGrid3x3:
    """
    Crop each glyph to its tight bounding box, then tile it into a 3×3 grid
    with minimal inter-glyph spacing — mimicking sequential character placement
    in a word or sentence rather than isolated padded tiles.

    h_gap:     horizontal pixels between columns (kerning approximation).
    v_gap:     vertical pixels between rows (leading approximation).
    peer_pool: optional list of PIL Images to draw surrounding cells from.
               If None, all 9 cells repeat the centre character.
               Peers are scaled to match the centre glyph height; bg colour is
               inverted when a peer's background differs from the centre's.
    """

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 peer_pool: list | None = None):
        self.h_gap     = h_gap
        self.v_gap     = v_gap
        self.padding   = padding
        self.peer_pool = peer_pool

    # ------------------------------------------------------------------
    def _crop_glyph(self, img: Image.Image) -> tuple[Image.Image, bool]:
        """Return (tight-cropped glyph, is_light_background)."""
        gray   = np.array(img.convert('L'))
        is_light = int(gray[0, 0]) > 128
        thr    = 200 if is_light else 50
        cmp    = (gray < thr) if is_light else (gray > thr)
        mask   = Image.fromarray(cmp.astype(np.uint8) * 255, 'L')
        bbox   = mask.getbbox() or (0, 0, img.width, img.height)
        return img.crop(bbox), is_light

    def __call__(self, img: Image.Image) -> Image.Image:
        center, center_light = self._crop_glyph(img)
        cw, ch = center.size
        bg_color = img.getpixel((0, 0))

        if self.peer_pool:
            peers = []
            for raw in random.choices(self.peer_pool, k=8):
                g, peer_light = self._crop_glyph(raw)
                if peer_light != center_light:
                    g = ImageOps.invert(g.convert('RGB')).convert(img.mode)
                scale = ch / g.height if g.height > 0 else 1.0
                g = g.resize((max(1, int(g.width * scale)), ch), Image.LANCZOS)
                peers.append(g)
        else:
            peers = [center] * 8

        # cells[4] is the centre; 0-3 are top-left quadrant, 5-8 bottom-right
        cells = peers[:4] + [center] + peers[4:]

        # per-column width = widest cell in that column
        col_w = [max(cells[r * 3 + c].width for r in range(3)) for c in range(3)]
        p = self.padding
        total_w = sum(col_w) + self.h_gap * 2 + p * 2
        total_h = ch * 3 + self.v_gap * 2 + p * 2
        grid = Image.new(img.mode, (total_w, total_h), bg_color)

        col_x = [p,
                 p + col_w[0] + self.h_gap,
                 p + col_w[0] + self.h_gap + col_w[1] + self.h_gap]
        for row in range(3):
            for col in range(3):
                cell = cells[row * 3 + col]
                x = col_x[col] + (col_w[col] - cell.width) // 2
                y = p + row * (ch + self.v_gap)
                grid.paste(cell, (x, y))
        return grid


class TileGrid3x3Rotated:
    """
    Builds the same tight 3×3 grid as TileGrid3x3, then rotates the ENTIRE
    grid by a randomly chosen angle from 8 evenly-spaced orientations
    (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°), simulating a full line of
    text captured at an angle or skew rather than rotating individual glyphs.
    Canvas expands to avoid clipping at diagonal angles.
    """
    _ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, h_gap: int = 2, v_gap: int = 4, padding: int = 6,
                 peer_pool: list | None = None):
        self._grid_tf = TileGrid3x3(h_gap=h_gap, v_gap=v_gap, padding=padding,
                                    peer_pool=peer_pool)

    def __call__(self, img: Image.Image) -> Image.Image:
        import math
        grid     = self._grid_tf(img)
        gw, gh   = grid.size
        bg_color = img.getpixel((0, 0))

        # Pre-compute the largest bounding box across all 8 angles so every
        # output is the same size regardless of which angle is chosen.
        max_w = max_h = 0
        for a in self._ANGLES:
            rad = math.radians(a)
            max_w = max(max_w, int(math.ceil(abs(gw * math.cos(rad)) + abs(gh * math.sin(rad)))))
            max_h = max(max_h, int(math.ceil(abs(gw * math.sin(rad)) + abs(gh * math.cos(rad)))))

        angle   = random.choice(self._ANGLES)
        rotated = grid.rotate(-angle, expand=True, fillcolor=bg_color)

        canvas = Image.new(img.mode, (max_w, max_h), bg_color)
        canvas.paste(rotated, ((max_w - rotated.width) // 2,
                                (max_h - rotated.height) // 2))
        return canvas


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


def get_transforms(augment: str = 'heavy', peer_pool: list | None = None):
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
            TileGrid3x3(peer_pool=peer_pool),
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


def build_dataset(dataset_dir, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
    root = Path(dataset_dir)
    all_class_names = sorted(d.name for d in root.iterdir() if d.is_dir())

    per_class = {}
    for cls in all_class_names:
        imgs = sorted((root / cls).glob('*.png'))
        if len(imgs) >= 5:
            per_class[cls] = imgs
        else:
            print(f'[data] skipping {cls!r}: only {len(imgs)} image(s)')

    class_names  = sorted(per_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_samples = [
        (p, class_to_idx[cls])
        for cls in class_names
        for p in per_class[cls]
    ]

    if not all_samples:
        raise ValueError(f'No training images found in {root}')

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

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sampler_tag = 'weighted' if weighted_sampler else 'shuffle'
    print(f'[data] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  |  {len(class_names)} classes  |  {sampler_tag} sampler')
    return train_loader, val_loader, test_loader, class_names
