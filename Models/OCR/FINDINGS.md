# OCR Char Classifier — Training Findings

## Latin Script: 60-Epoch Colab Run

**Config:** DINOv2 ViT-S/14, 62 classes (26 cap + 26 low + 10 digit), batch 32, heavy augment, grid-mode single, mixup α=0.4, freeze-epochs 5, unfreeze-blocks 4, LR 1e-3, CosineAnnealingLR over fine-tune phase.
**Dataset:** 77,799 images (~1,255/class avg). Val split 10%.
**Hardware:** Colab GPU. Total run time: 16.4 h (avg 984 s/epoch).

### Results

| Metric | Value | Epoch |
|---|---|---|
| Best val accuracy | **55.6%** | 42 |
| Best val F1 (macro) | **59.8%** | 42 |
| Val precision | ~67–69% | — |
| Val recall | ~54% | — |
| Final val accuracy | 54.2% | 60 |
| Random baseline | 1.6% (1/62) | — |

### Phase Analysis

**Phase 1 — head warm-up (epochs 1–5, backbone frozen)**
- Peak val acc: **15.5%**. Contributes almost nothing.
- All 5 epochs waste ~4,360 s (~1.2 h) of compute.

**Phase 2 — backbone fine-tune (epochs 6–60)**
- Epoch 6 (first unfreeze epoch): val acc jumps from 15.5% → **45.8%** in a single step. The backbone does everything.
- Gradual climb epochs 6–42, peak at **55.6%** val acc with LR ≈ 6e-5.
- Val acc plateaus and slowly regresses after epoch 42. LR continues decaying to ~0.

**Dead tail (epochs 56–60)**
- LR drops below 1e-7 at epoch 56 (cosine schedule effectively exhausted).
- Val acc frozen at 54.2%, zero improvement across 5 epochs.
- These epochs waste ~4,600 s with no benefit.

### Observations

- **Train acc < Val acc throughout** — expected with MixUp α=0.4; blended training images are genuinely harder to classify than clean val images. Not a bug.
- **Precision–recall gap** — precision ~68% but recall ~54%. Model is conservative: confident when it commits, but under-predicts less common classes. More images per class would help recall more than more epochs.
- **Cosine schedule exhaustion is the ceiling**, not the model capacity. The model was still improving at epoch 42 when the LR reached ~6e-5; the decay to zero caused the plateau, not saturation.

---

## Implementation Plan

### 1 — Reduce freeze epochs: 5 → 3

Phase 1 is overhead. The backbone unfreeze at epoch 6 delivers a +30 pt jump regardless of how long the head warmed up. Cutting to 3 saves ~1,730 s per run with no accuracy cost.

```
--freeze-epochs 3   # was 5
```

### 2 — Shorten epoch budget: 60 → 48

Best epoch was 42; the tail (43–60) yielded 0 gain. Budget 48 epochs to allow a small overshoot window past the expected peak without wasting 5+ dead epochs at LR ≈ 0.

```
--epochs 48   # was 60 (saves ~2 h on Colab)
```

### 3 — Add warm-restart scheduler (`--scheduler` flag)

`CosineAnnealingLR` decays to zero and stays there. `CosineAnnealingWarmRestarts` resets the LR periodically, giving the model repeated chances to escape local minima. Add a `--scheduler` flag to `train.py`:

```
--scheduler cosine       # current default (CosineAnnealingLR)
--scheduler cosine-warm  # CosineAnnealingWarmRestarts(T_0=15, T_mult=2)
--scheduler none         # constant LR (useful for phase 1 debugging)
```

With T_0=15 and T_mult=2: restarts at epochs 6+15=21, 21+30=51 — fits naturally in a 48-epoch phase-2 window.

### 4 — Lower MixUp alpha: 0.4 → 0.2

α=0.4 creates a consistent 10–15 pt train/val gap and may be hurting convergence speed. α=0.2 keeps regularization while producing less extreme blends. Measure val F1 on next run to confirm before locking in.

```
--mixup-alpha 0.2   # was 0.4
```

### Implementation target: `Models/OCR/char_classifier/train.py`

The `--scheduler` flag requires changes to `_make_p2_optimizer` / the phase-2 `train_loop` call site. The other three are arg-default changes only.

---

## CJK / All-Scripts Analysis

### Dataset comparison

| Script | Classes | Images | Avg/class | Val set size |
|---|---|---|---|---|
| Latin | 62 | 77,799 | ~1,255 | ~7,780 |
| Kana | 169 | ~2,036 | ~12 | ~200 |
| Hangul | 500 | 6,000 | 12 | ~600 |
| CJK | 1,312 | ~7,354 | ~5.6 | ~735 |

### Expected behavior differences

**Phase 1 (freeze):** Same conclusion — backbone frozen contributes nothing. Keep freeze-epochs at 3 for all scripts.

**Phase 2 jump magnitude:** The +30 pt backbone-unfreeze jump seen in Latin should also appear for other scripts — DINOv2 features are strongly transferable. However, the ceiling will be lower because of data sparsity.

**Overfitting risk:** Latin never showed overfitting (train acc stayed below val acc due to MixUp). CJK and Hangul have ~5-12 images/class — classic few-shot regime. With MixUp at α=0.2 and the weighted sampler, the risk is manageable but real. **Monitor overfit_gap in progress.json**; if it flips positive and grows, reduce epochs or add dropout.

**Epoch timing:** CJK and Hangul have far fewer total images than Latin, so epochs will be much faster despite more classes. Estimated:
- Kana: ~30–60 s/epoch on Colab (vs 984 s for Latin)
- Hangul: ~100–200 s/epoch
- CJK: ~120–250 s/epoch
- All scripts: dominated by Latin — roughly ~1,100 s/epoch (adds Kana/Hangul/CJK overhead)

**Val set reliability:** With ~5 images/class avg for CJK, val set gets ~0–1 image per class. Val accuracy will be very noisy epoch-to-epoch. Best epoch may not be well-defined. Mitigations: lower `--min-per-class` only if a class has ≥3 images; don't over-rely on single-epoch val acc as stopping criterion.

**Warm restarts matter more for small datasets.** With 5-12 images/class, the loss surface is rougher and the model is more likely to get stuck in local minima. `--scheduler cosine-warm` should be the default for Kana/Hangul/CJK.

### Recommended run configs per script

```powershell
# Kana — fast, small dataset, low risk; cosine-warm to escape local minima
.venv\Scripts\python.exe -m char_classifier.train \
    --scripts kana --epochs 48 --freeze-epochs 3 \
    --scheduler cosine-warm --mixup-alpha 0.2

# Hangul — same rationale as Kana
.venv\Scripts\python.exe -m char_classifier.train \
    --scripts hangul --epochs 48 --freeze-epochs 3 \
    --scheduler cosine-warm --mixup-alpha 0.2

# CJK — few-shot, expect noisy val; longer budget OK since epochs are cheap
.venv\Scripts\python.exe -m char_classifier.train \
    --scripts cjk --epochs 60 --freeze-epochs 3 \
    --scheduler cosine-warm --mixup-alpha 0.2

# All scripts (single combined model) — Latin dominates timing
# Useful for compare.py but per-script models are preferred for deployment
.venv\Scripts\python.exe -m char_classifier.train \
    --scripts all --epochs 48 --freeze-epochs 3 \
    --scheduler cosine-warm --mixup-alpha 0.2
```

### Per-script vs combined model

The existing architecture supports both. Per-script models are preferred:
- The routing logic in `ocr_pipeline.py` already dispatches crops to the correct script's model
- A combined model would be dominated by Latin (80% of images) at the expense of CJK/Hangul accuracy
- Each script can be trained and updated independently
