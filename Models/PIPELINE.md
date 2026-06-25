# Models Pipeline — Status & Next Steps

_Last updated: 2026-06-25_

---

## Directory Layout

```
Models/
  Datasets/             ← data generators (shared)
    get_fonts.py        ← scan Windows fonts → font-dataset/ (font classifier)
    render_chars.py     ← render chars → char-dataset/{latin,kana,hangul,cjk}/
    sample_tilegrid*.py ← visual sanity checks for grid augments
    char-dataset/
      latin/            ← 62 classes, 77,799 images
      kana/             ← 172 classes (169 viable), 2,036 images
      hangul/           ← 500 classes, 6,000 images
      cjk/              ← 3,000 classes (1,312 viable), 10,506 images
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

### 1. Datasets — COMPLETE (all 4 scripts)

| Script  | Generator    | Output dir                    | Class dirs | Viable (≥5 img) | Images |
|---------|-------------|-------------------------------|------------|-----------------|--------|
| Latin   | render_chars | char-dataset/latin/           | 62         | 62              | 77,799 |
| Kana    | render_chars | char-dataset/kana/            | 172        | 169             | 2,036  |
| Hangul  | render_chars | char-dataset/hangul/          | 500        | 500             | 6,000  |
| CJK     | render_chars | char-dataset/cjk/             | 3,000      | 1,312           | 10,506 |
| Fonts   | get_fonts    | font-dataset/                 | —          | —               | 21,676 |

Paths are anchored to `Models/Datasets/` via `_HERE = Path(__file__).parent`
so both scripts work from any cwd.

**Render parameters:** 2 sizes (32 px / 96 px), 2 modes (light/dark) → 4 images per font per class. Design decision: more than 2 sizes does not meaningfully improve recognition quality.

**CJK coverage gap:** 1,688 of 3,000 CJK class dirs have zero images — no Windows system fonts cover those codepoints. The remaining 1,312 viable classes each have ≥2 Windows fonts (≥8 images). The gap is not fixable by changing render sizes; it requires fonts with broader CJK coverage (e.g. Noto CJK).

**Extending CJK coverage (when Noto fonts are available):**
```powershell
cd Models\Datasets
python render_chars.py --scripts cjk --cjk-top 3000 --extra-fonts-dir C:\path\to\noto-downloads
```
The `--extra-fonts-dir` flag is already implemented — fonts are used in-place, not copied to `windows-fonts/`. After re-render, pass `--min-per-class 1` to train on lower-coverage classes if desired.

**Kana note:** 3 of 172 classes are empty (U+3094/3095/3096 — obsolete kana absent from Windows fonts). 169 viable classes with ≥5 images.

**render_chars.py CLI reference:**
```powershell
python render_chars.py --scripts latin
python render_chars.py --scripts kana
python render_chars.py --scripts hangul --hangul-top 500
python render_chars.py --scripts cjk    --cjk-top 3000
python render_chars.py --scripts cjk    --cjk-top 3000 --extra-fonts-dir C:\path\to\noto
python render_chars.py --scripts all    --hangul-top 500 --cjk-top 3000
```

---

### 2. OCR / char_classifier — DATASET READY, UNTRAINED

**Architecture:** DINOv2 ViT-S/14 backbone + linear head, two-phase training
(head warm-up → backbone fine-tune).

**Supported scripts:** Latin (62 classes), Kana (169 viable), Hangul (500), CJK (1,312 viable).
Train per-script for scoped checkpoints; train `--scripts all` for a unified classifier.

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

# Quick smoke test (CPU, any script)
.venv\Scripts\python.exe -m char_classifier.train --scripts latin --epochs 5 --freeze-epochs 5 --max-per-class 20

# Per-script full runs (GPU recommended)
.venv\Scripts\python.exe -m char_classifier.train --scripts latin  --epochs 30 --freeze-epochs 5 --grid-mode all --backbone dinov2_vits14 --batch-size 64
.venv\Scripts\python.exe -m char_classifier.train --scripts kana   --epochs 30 --freeze-epochs 5 --grid-mode all --backbone dinov2_vits14 --batch-size 64
.venv\Scripts\python.exe -m char_classifier.train --scripts hangul --epochs 30 --freeze-epochs 5 --grid-mode all --backbone dinov2_vits14 --batch-size 64
.venv\Scripts\python.exe -m char_classifier.train --scripts cjk    --epochs 30 --freeze-epochs 5 --grid-mode all --backbone dinov2_vits14 --batch-size 64

# Resume from checkpoint
.venv\Scripts\python.exe -m char_classifier.train --scripts latin --resume checkpoints/latin/best.pt

# Lower class threshold (e.g. after Noto re-render adds more classes)
.venv\Scripts\python.exe -m char_classifier.train --scripts cjk --min-per-class 1
```

Checkpoints auto-scope to `checkpoints/<script>/` for single-script runs.

**Training state:** No checkpoint exists yet for any script — never been trained.

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

CJK recognition continues to use EasyOCR's built-in CJK readers until
char_classifier is trained and validated on CJK scripts. The CRAFT detector is
script-agnostic — it finds all text regions regardless of script.

---

## Next Steps (priority order)

### Immediate
1. **Train char_classifier (Latin)** — 62 classes, dataset ready. Start with
   `--grid-mode single` for a baseline, then compare `--grid-mode all`.
   GPU strongly recommended (CPU was ~3.7 s/iter on DINOv2 ViT-S).

2. **Wire CRAFT + char_classifier together** — create
   `Models/OCR/ocr_pipeline.py` that calls `detect_and_crop` → `predict`
   and returns `[(text, confidence, bbox)]` matching EasyOCR's output format.
   This replaces EasyOCR for Latin-script regions.

### Near-term
3. **Train char_classifier (Kana / Hangul / CJK)** — datasets ready.
   Kana and Hangul are well-covered. CJK has 1,312 viable classes (10,506 images)
   using Windows fonts — usable for a baseline. See below for expansion.

4. **Expand CJK dataset via Noto fonts** — download Noto Sans/Serif CJK
   (`NotoSansCJK-Regular.ttc`), then re-render:
   ```powershell
   python render_chars.py --scripts cjk --cjk-top 3000 --extra-fonts-dir C:\path\to\noto
   ```
   This unlocks the 1,688 zero-coverage classes (all within U+4E00–U+9FFF)
   and increases per-class image count for the existing 1,312 classes.
   Infrastructure (`--extra-fonts-dir`, `--min-per-class`) already in place.

5. **Evaluate char_classifier on real Discord crops** — test with actual
   screenshots from `Translation/0-Data/Image/data/` using `sample_craft.py
   --image <path>` to exercise the full detect → crop path.

6. **peer_pool integration** — `get_dataloaders` accepts `peer_pool` but it
   is always passed as `None` today. Pre-loading all training images as a peer
   pool for `TileGrid3x3` / `TileGrid3x3Pair` would enable cross-character
   context grids (the Y-neighbour strategy from README.md).

### Longer-term
7. **Train font_classifier** — 184 classes, baseline done. GPU run with 30
   epochs + phase 2 backbone fine-tune needed.

8. **Replace EasyOCR in TL-Bot.py** — once char_classifier is trained and
   validated, swap the CRAFT + char_classifier pipeline into
   `Translation/2-Image/ocr.py` as an alternative recognizer path.

9. **Korean routing fix in ocr_pipeline.py** — manga-ocr maps Korean crops
   to Japanese; post-hoc Hangul check misses it. Needs EasyOCR-Korean parallel
   pass or a script pre-classifier trained on char_classifier Hangul output.
