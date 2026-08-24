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

---

## Latin Tile-Context Investigation (Runs 3-6)

The 60-epoch run above trained on the v1 isolated-tile pipeline — centered single-glyph tiles, each additionally passed through `TileGrid3x3` synthetic augmentation (the character rendered into a 3x3 grid of distractor tiles to force the model to localize the target glyph rather than assume it fills the frame). That combination hit a hard ceiling around 55%, and manual error inspection showed a specific failure mode: ambiguous single-stroke fragments (`low_i`, `low_l`, `cap_I`, digit `1`) collapsing into whichever of those classes was statistically dominant, rather than being distinguished on their actual shape. Runs 3-6 isolated and fixed this.

**Naming note:** at the time these runs were executed, the isolated-tile dataset was named `char-dataset` and the real-context dataset was named `char-dataset-ctx-small`. Following run 6's confirmation, both were renamed to reflect the new pipeline as the default: the real-context dataset is now plain `char-dataset`, and the old isolated-tile dataset is archived as `char-dataset-legacy` (generated by `render_chars.py`, kept for retrieval against the archived run 1-4 checkpoint only, not regenerated). The table below uses the names as they exist now, post-rename.

### Hypothesis

`TileGrid3x3` synthesizes visual context by embedding an isolated glyph tile among distractor tiles cropped from the *same* isolated-glyph pool. This never resembles real running text — no shared baseline, no natural inter-character spacing/kerning, no consistent stroke width across the "context." For thin single-stroke glyphs, the model had no reliable real-world cue to key off, so it fell back on class-frequency priors.

The fix under test: `render_chars_context.py` generates a fundamentally different dataset by rendering real strings and cropping to the target glyph *with its true neighboring characters still in frame* — real baseline, real kerning, real stroke-width consistency. The question across runs 3-6 was whether this fixes the collapse, and whether stacking synthetic `TileGrid3x3` on top of already-real context helps, hurts, or is neutral.

### Results

| Run | Dataset | `grid_mode` | Best val acc | Epoch | `low_i`-involved errors |
|---|---|---|---|---|---|
| 3 | `char-dataset-legacy` (isolated tiles) | `single` (TileGrid3x3) | 51.2% | 10 | — (baseline collapse pattern) |
| 4 | `char-dataset-legacy` (isolated tiles, widened crop scale) | `single` | 48.1% | 15 | — (crop-scale-only tuning; did not fix, only relocated the collapse) |
| 5 | `char-dataset` (real string context) | `single` (TileGrid3x3 stacked on top) | 77.2% | 24 | 17.5% of all errors (97/555) |
| **6** | `char-dataset` (real string context) | **`none`** (no grid tiling) | **90.2%** | 23 | **5.2% of all errors (13/248)** |

Run 6 config: DINOv2 ViT-S/14, `freeze_epochs=3`, `epochs=24`, dataset `char-dataset` (named `char-dataset-ctx-small` at run time), `grid_mode=none`. Checkpoint: `latin_ctx-small/best.pt` (epoch 23) on the Drive mount — the Drive checkpoint path retains its original `latin_ctx-small` name; only the local dataset folder and default `train.py`/`remote_train.py` args were renamed.

### Conclusions

1. **Real string-rendered context (runs 5 vs 3/4) fixes most of the collapse on its own** — moving from isolated tiles to real-context crops took best val acc from ~51% to ~77%, a +26pt jump, before touching the grid-tiling question at all.
2. **Stacking `TileGrid3x3` on top of already-real context is actively harmful, not neutral** (run 6 vs run 5, same dataset, same seed/epoch budget, only `grid_mode` differs). Removing the redundant synthetic tiling produced a further +13pt jump (77.2% -> 90.2%) and, more importantly, cut the `low_i`-class stroke-fragment error rate from 17.5% to 5.2% of all errors — the specific pathology this investigation set out to fix. This is not just a higher baseline with the same error profile; the targeted mechanism improved directly.
3. **Residual errors in run 6 are ordinary case/glyph-shape ambiguity** (`cap_S`/`low_s`, `cap_I`/`low_l`, `dig_0`/`cap_O`, `low_v`/`cap_V`, `low_w`/`cap_W`, etc.) — pairs that are genuinely hard to distinguish from a single cropped glyph regardless of pipeline quality. Rotation-ambiguous confusions are negligible (1 error, 0.4% of total). No new failure mode was introduced by removing grid tiling.
4. **Takeaway for future dataset design**: synthetic context augmentation is not a substitute for real context, and applying it on top of real context can actively degrade the signal the model would otherwise learn from genuine kerning/baseline/stroke-width cues. `grid_mode=none` should be the default whenever the underlying tiles already carry real string context (i.e. whenever `dataset_name` points at a `render_chars_context.py`-generated dataset); `TileGrid3x3` variants remain appropriate only for isolated-glyph datasets like the legacy `char-dataset-legacy`.

**Promoted as the unqualified default** (see `train.py` / `remote_train.py`, no flags needed): `dataset_name=char-dataset` (real-context pipeline), `grid_mode=none`. The legacy isolated-tile pipeline is now opt-in only: `--dataset-name char-dataset-legacy` (with `render_chars.py`, not regenerated by default) plus an explicit non-`none` `--grid-mode` if you want to reproduce runs 3/4's tiling behavior.

### Next training items

- **Validate at full scale**: run 6 used the downsampled real-context dataset. Confirm whether training on the full uncapped variant (`char-dataset-ctx`, still present locally, not yet promoted) yields further gains, or whether the current default already captures the available signal — worth one comparison run before assuming bigger is better here.
- **Extend the real-context approach to kana/hangul/cjk.** These scripts are far more data-sparse (12, 12, and 5.6 images/class avg respectively vs Latin's ~1,255) and currently still train on isolated tiles with `TileGrid3x3`. If the same collapse mechanism applies (likely, given how general the underlying cause is — synthetic distractor context vs. real kerning/baseline cues), the fix should transfer, but few-shot regimes may behave differently. Needs `render_chars_context.py` extended to those scripts' character sets first, plus font coverage broad enough to render real strings (font-diversity expansion, e.g. Noto CJK/Sans-KR, is a listed prerequisite).
- **Re-run `compare_preprocess.py` / `compare.py` end-to-end** against the new Latin checkpoint once deployed, to confirm the accuracy gain holds up through the full OCR pipeline (CRAFT detection -> char_classifier), not just on the held-out val split.
- **Revisit `--epochs` budget for the new dataset/grid_mode combination.** Run 6's best epoch was 23 of 24 (val_acc still rising, `epochs_since_best=1` at the final epoch) — unlike the original 60-epoch Latin run, which plateaued and had a wasted dead tail, run 6 may have been cut off before full convergence. A longer run (e.g. 32-40 epochs) is worth trying to see if val acc climbs further before plateauing.
