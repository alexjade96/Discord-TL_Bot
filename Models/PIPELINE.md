# Models Pipeline — Status & Next Steps

_Last updated: 2026-06-23_

---

## Directory Layout

```
Models/
  Datasets/             ← data generators (shared)
    get_fonts.py        ← scan Windows fonts → font-dataset/ (font classifier)
    render_chars.py     ← render chars → char-dataset/latin/ (char classifier)
    sample_tilegrid*.py ← visual sanity checks for grid augments
    char-dataset/latin/ ← 62 classes, 77,799 images (render_chars output)
    font-dataset/       ← 21,676 images (get_fonts output)

  OCR/
    grid_augments.py                  ← 6 grid transforms + RandomGridAugment
    char_classifier/                  ← DINOv2 character recognizer
      data.py / train.py / engine.py  ← full training pipeline
      model_builder.py / predict.py   ← inference
    detection/                        ← CRAFT text detection wrapper
      craft_detector.py               ← detect() / detect_and_crop() API
      sample_craft.py                 ← end-to-end test script
      sample_craft_source.png         ← generated test image
      sample_craft_annotated.png      ← 8 regions correctly detected

  Typography/
    font_classifier/   ← DINOv2 font family classifier (9 modules, mirrors OCR)
    Font_Classifier.ipynb
    train_2epoch.log   ← baseline run result
```

---

## Component Status

### 1. Datasets — COMPLETE

| Generator     | Output dir                         | Classes | Images  |
|---------------|------------------------------------|---------|---------|
| render_chars  | Datasets/char-dataset/latin/       | 62      | 77,799  |
| get_fonts     | Datasets/font-dataset/             | —       | 21,676  |

Paths are anchored to `Models/Datasets/` via `_HERE = Path(__file__).parent`
so both scripts work from any cwd.

---

### 2. OCR / char_classifier — DATASET READY, UNTRAINED

**Architecture:** DINOv2 ViT-S/14 backbone + linear head, two-phase training
(head warm-up → backbone fine-tune).

**Dataset:** 62 Latin character classes, 77,799 synthetic images rendered across
multiple fonts, sizes (32 px / 96 px), and modes (dark/light).

**Augmentation pipeline (heavy mode):**
```
SimulateJPEG(q=40-90)
→ RandomAffine(shear=10)
→ RandomPerspective(distortion=0.2, p=0.3)
→ [Grid augment — see below]
→ RandomResizedCrop(224, scale=0.20-0.45)
→ RandomRotation(±12°)
→ ColorJitter(brightness/contrast=0.4)
→ GaussianBlur(σ=0.1-2.0)
→ AddGaussianNoise(std=0.04)
→ RandomErasing(p=0.2)
→ Normalize(ImageNet stats)
```

**Grid augment modes (`--grid-mode`):**

| Mode      | Transforms active                                                   |
|-----------|---------------------------------------------------------------------|
| `single`  | TileGrid3x3 only (default, fastest)                                 |
| `rotated` | Random choice: TileGrid3x3 or TileGrid3x3Rotated per sample         |
| `all`     | Random choice among all 6 variants per sample                       |

All 6 variants: TileGrid3x3, TileGrid3x3Rotated, TileGrid3x3Pair,
TileGrid3x3PairRotated, TileGrid3x3Orbital, TileGrid3x3OrbitalRotated.

**Training commands:**
```powershell
cd Models/OCR
# Quick smoke test (CPU, 5 epochs, head only)
.venv\Scripts\python.exe -m char_classifier.train --epochs 5 --freeze-epochs 5

# Full run (GPU recommended, all grid variants)
.venv\Scripts\python.exe -m char_classifier.train \
    --epochs 30 --freeze-epochs 5 --grid-mode all \
    --backbone dinov2_vits14 --batch-size 64

# Resume from checkpoint
.venv\Scripts\python.exe -m char_classifier.train --resume checkpoints/best.pt
```

**Training state:** No checkpoint exists yet — never been trained.

---

### 3. OCR / detection (CRAFT) — COMPLETE

**Status:** Fully implemented and tested end-to-end.

| File                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| craft_detector.py     | `detect()` / `detect_and_crop()` — public API        |
| sample_craft.py       | Test harness; generates synthetic Discord image      |

**Result of last test run (2026-06-23):**
- Synthetic 640×320 Discord dark-mode image with 8 text lines
- 8/8 regions correctly detected (English, Korean, Japanese, Chinese, usernames)
- Weights cached at `~/.craft_text_detector/weights/`
- `poly=False` required — NumPy ≥1.24 breaks polygon mode

**Known compatibility patches applied:**
- `.venv/Lib/.../craft_text_detector/models/basenet/vgg16_bn.py` — replaced
  removed `model_urls` with `VGG16_BN_Weights.DEFAULT`
- Install must use `--no-deps` + `pip install gdown` separately

**Usage:**
```python
from detection.craft_detector import detect, detect_and_crop

boxes  = detect("screenshot.png")                   # [(x1,y1,x2,y2), ...]
crops  = detect_and_crop("screenshot.png", pad=4)   # [PIL.Image, ...]
```

---

### 4. Typography / font_classifier — BASELINE RUN DONE, NEEDS TRAINING

**Architecture:** Identical structure to char_classifier (DINOv2 ViT-S/14,
two-phase, MixUp, CosineAnnealingLR).

**Baseline result (2 epochs, head warm-up only, CPU):**
```
Epoch 1 | train acc 0.0096 | val acc 0.0250  top3 0.0362  top5 0.0459
Epoch 2 | train acc 0.0126 | val acc 0.0278  top3 0.0403  top5 0.0662
Test   top-1 0.0265  top-3 0.0440  top-5 0.0699
```
Expected — 2 epochs on CPU with 184 classes barely moves the head. Needs a
proper GPU run with full phase 2 fine-tune to be useful.

**Training commands:**
```powershell
cd Models/Typography
.venv\Scripts\python.exe -m font_classifier.train \
    --epochs 30 --freeze-epochs 5 \
    --backbone dinov2_vits14 --batch-size 64
```

---

## Integration Picture

```
Discord screenshot
        │
        ▼
  craft_detector.detect()          ← Models/OCR/detection/
        │  boxes [(x1,y1,x2,y2)]
        ▼
  detect_and_crop()                ← crop each region
        │  [PIL.Image, ...]
        ▼
  char_classifier.predict()  ──OR──  EasyOCR recognizer
        │  class labels / strings
        ▼
  translate_text()                 ← Translation/1-Text/translate_text.py
        │
        ▼
  Discord response
```

The char_classifier is a **Latin-only** recognizer (A-Z, a-z, 0-9, punctuation).
CJK recognition continues to use EasyOCR's built-in CJK readers.
The CRAFT detector is script-agnostic — it finds all text regions regardless of script.

---

## Next Steps (priority order)

### Immediate
1. **Train char_classifier** — 62 classes, dataset ready. Start with
   `--grid-mode single` for a baseline, then compare `--grid-mode all`.
   GPU strongly recommended (CPU was ~3.7 s/iter on DINOv2 ViT-S).

2. **Wire CRAFT + char_classifier together** — create
   `Models/OCR/ocr_pipeline.py` that calls `detect_and_crop` → `predict`
   and returns `[(text, confidence, bbox)]` matching EasyOCR's output format.
   This replaces EasyOCR for Latin-script regions.

### Near-term
3. **Train font_classifier** — 184 classes, baseline done. GPU run with 30
   epochs + phase 2 backbone fine-tune needed.

4. **Evaluate char_classifier on real Discord crops** — test with actual
   screenshots from `Translation/0-Data/Image/data/` using `sample_craft.py
   --image <path>` to exercise the full detect → crop path.

5. **peer_pool integration** — `get_dataloaders` accepts `peer_pool` but it
   is always passed as `None` today. Pre-loading all training images as a peer
   pool for `TileGrid3x3` / `TileGrid3x3Pair` would enable cross-character
   context grids (the Y-neighbour strategy from README.md).

### Longer-term
6. **CJK character dataset** — `render_chars.py` currently renders Latin only.
   Extend to Hangul, CJK Unified Ideographs, Hiragana/Katakana for a
   unified char_classifier that covers all scripts the bot handles.

7. **Replace EasyOCR in TL-Bot.py** — once char_classifier is trained and
   validated, swap the CRAFT + char_classifier pipeline into
   `Translation/2-Image/ocr.py` as an alternative recognizer path.
