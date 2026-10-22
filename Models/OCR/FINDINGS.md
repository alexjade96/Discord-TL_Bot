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

## Latin Tile-Context Investigation (Runs 3-7)

The 60-epoch run above trained on the v1 isolated-tile pipeline — centered single-glyph tiles, each additionally passed through `TileGrid3x3` synthetic augmentation (the character rendered into a 3x3 grid of distractor tiles to force the model to localize the target glyph rather than assume it fills the frame). That combination hit a hard ceiling around 55%, and manual error inspection showed a specific failure mode: ambiguous single-stroke fragments (`low_i`, `low_l`, `cap_I`, digit `1`) collapsing into whichever of those classes was statistically dominant, rather than being distinguished on their actual shape. Runs 3-6 isolated and fixed this; run 7 validated the fix at full dataset scale and settled the epoch-budget question.

**Naming note:** at the time these runs were executed, the isolated-tile dataset was named `char-dataset` and the real-context dataset was named `char-dataset-ctx-small`. Following run 6's confirmation, both were renamed to reflect the new pipeline as the default: the real-context dataset is now plain `char-dataset`, and the old isolated-tile dataset is archived as `char-dataset-legacy` (generated by `render_chars.py`, kept for retrieval against the archived run 1-4 checkpoint only, not regenerated). The table below uses the names as they exist now, post-rename. Run 7 uses `char-dataset-ctx` — the full/uncapped real-context variant, distinct from the downsampled `char-dataset` / run-time `char-dataset-ctx-small` that runs 5-6 used.

### Hypothesis

`TileGrid3x3` synthesizes visual context by embedding an isolated glyph tile among distractor tiles cropped from the *same* isolated-glyph pool. This never resembles real running text — no shared baseline, no natural inter-character spacing/kerning, no consistent stroke width across the "context." For thin single-stroke glyphs, the model had no reliable real-world cue to key off, so it fell back on class-frequency priors.

The fix under test: `render_chars_context.py` generates a fundamentally different dataset by rendering real strings and cropping to the target glyph *with its true neighboring characters still in frame* — real baseline, real kerning, real stroke-width consistency. The question across runs 3-6 was whether this fixes the collapse, and whether stacking synthetic `TileGrid3x3` on top of already-real context helps, hurts, or is neutral. Run 7 then asked two follow-ups left open by run 6: does the full uncapped dataset beat the downsampled one, and was run 6's 24-epoch budget cut off before the true peak.

### Results

| Run | Dataset | `grid_mode` | Best val acc | Epoch | `low_i`-involved errors |
|---|---|---|---|---|---|
| 3 | `char-dataset-legacy` (isolated tiles) | `single` (TileGrid3x3) | 51.2% | 10 | — (baseline collapse pattern) |
| 4 | `char-dataset-legacy` (isolated tiles, widened crop scale) | `single` | 48.1% | 15 | — (crop-scale-only tuning; did not fix, only relocated the collapse) |
| 5 | `char-dataset` (real string context) | `single` (TileGrid3x3 stacked on top) | 77.2% | 24 | 17.5% of all errors (97/555) |
| 6 | `char-dataset` (real string context, downsampled) | `none` (no grid tiling) | 90.2% | 23 | 5.2% of all errors (13/248) |
| **7** | `char-dataset-ctx` (real string context, full/uncapped) | **`none`** | **91.1%** | 20 | **5.9% of all errors (13/219)** |

Run 6 config: DINOv2 ViT-S/14, `freeze_epochs=3`, `epochs=24`, dataset `char-dataset` (named `char-dataset-ctx-small` at run time), `grid_mode=none`. Checkpoint: `latin_ctx-small/best.pt` (epoch 23) on the Drive mount — the Drive checkpoint path retains its original `latin_ctx-small` name; only the local dataset folder and default `train.py`/`remote_train.py` args were renamed.

Run 7 config: same as run 6 except `epochs=36`, `mixup_alpha=0.2`, `batch_size=64`, and dataset `char-dataset-ctx` (full/uncapped real-context variant, ~155k Latin images / 62 classes vs run 6's downsampled pool). `git_commit=be3e28c6ad`. Checkpoint: `latin_ctx/best.pt` (epoch 20) on the Drive mount. Best val_acc **0.911438 @ epoch 20**; epochs 21-36 were all below that peak (range 0.9066-0.9107, `epochs_since_best=16` at the final epoch), LR decayed to 0.0 on the cosine schedule with no further improvement. One benign duplicate `epoch: 7` entry in `progress.json` `history` from a single early Colab disconnect/resume; top-level accounting stayed correct (`completed: 36`) and `best.pt` selection was unaffected.

### Conclusions

1. **Real string-rendered context (runs 5 vs 3/4) fixes most of the collapse on its own** — moving from isolated tiles to real-context crops took best val acc from ~51% to ~77%, a +26pt jump, before touching the grid-tiling question at all.
2. **Stacking `TileGrid3x3` on top of already-real context is actively harmful, not neutral** (run 6 vs run 5, same dataset, same seed/epoch budget, only `grid_mode` differs). Removing the redundant synthetic tiling produced a further +13pt jump (77.2% -> 90.2%) and, more importantly, cut the `low_i`-class stroke-fragment error rate from 17.5% to 5.2% of all errors — the specific pathology this investigation set out to fix. This is not just a higher baseline with the same error profile; the targeted mechanism improved directly.
3. **Residual errors in run 6 are ordinary case/glyph-shape ambiguity** (`cap_S`/`low_s`, `cap_I`/`low_l`, `dig_0`/`cap_O`, `low_v`/`cap_V`, `low_w`/`cap_W`, etc.) — pairs that are genuinely hard to distinguish from a single cropped glyph regardless of pipeline quality. Rotation-ambiguous confusions are negligible (1 error, 0.4% of total). No new failure mode was introduced by removing grid tiling.
4. **Takeaway for future dataset design**: synthetic context augmentation is not a substitute for real context, and applying it on top of real context can actively degrade the signal the model would otherwise learn from genuine kerning/baseline/stroke-width cues. `grid_mode=none` should be the default whenever the underlying tiles already carry real string context (i.e. whenever `dataset_name` points at a `render_chars_context.py`-generated dataset); `TileGrid3x3` variants remain appropriate only for isolated-glyph datasets like the legacy `char-dataset-legacy`.

### Run 7 conclusions (full-scale validation + epoch budget)

5. **The full uncapped dataset helps, but only marginally.** Run 7 on `char-dataset-ctx` (~155k images, uncapped) reached best val_acc **0.911438** vs run 6's **0.9023** on the downsampled variant — a **+0.92 pt** gain, same config otherwise (bar `mixup_alpha` 0.2 vs 0.4 and `batch_size` 64 vs 32). On the confused-pairs sample the total error count dropped 248 → 219 (−11.7%) and sample top-1 rose 0.9008 → 0.9124. Real but not a step change; run 6's downsampled variant already captured most of the available signal. The uncapped variant is worth using going forward, not required.
6. **Run 6 was NOT cut off before its peak.** Run 7 had a 36-epoch budget and **peaked at epoch 20**, then flatlined for 16 epochs (`epochs_since_best: 16` at epoch 36; every epoch 21-36 below the epoch-20 value; closest was epoch 27 at 0.9107). The earlier worry that run 6's 24-epoch budget was short is resolved: for this dataset+config val_acc is done improving by ~epoch 20-23. The FINDINGS note that "run 6 gained most of its late accuracy during the cosine decay tail" holds for run 6's shorter schedule but does not generalize — with a longer schedule the gains arrive earlier and then stop. **Practical implication: budget ~24-28 epochs for future Latin runs on this config, not 36.**
7. **The targeted `low_i` stroke-fragment pathology holds at its run-6 floor — no regression.** Run 7's confused-pairs diagnostic (2,500-sample, seed 42, `confused_pairs_latin_run7.json`) shows **13** `low_i`-involved errors, identical in absolute count to run 6. The fraction ticked 5.24% → 5.94% only because the denominator (total errors) shrank; the pathology itself did not worsen. The run-6 investigation drove this from 17.5% (run 5) to 5.2%; run 7 keeps it there. The error *character* softened slightly (run 6: `low_i→low_l` ×4, `low_i→dig_1` ×4; run 7: `low_i→low_j` ×6 — i/j tittle+descender ambiguity is a less damaging confusion than i/1). Rotation-ambiguous confusions dropped from 1 to 0. The one visibly worse pair is `cap_I→low_l` (13 → 19), which is ordinary case ambiguity in sans fonts, not the crop-fragment mechanism.

**Promoted as the unqualified default** (see `train.py` / `remote_train.py`, no flags needed): `dataset_name=char-dataset` (real-context pipeline), `grid_mode=none`. The legacy isolated-tile pipeline is now opt-in only: `--dataset-name char-dataset-legacy` (with `render_chars.py`, not regenerated by default) plus an explicit non-`none` `--grid-mode` if you want to reproduce runs 3/4's tiling behavior.

**Promoted Latin checkpoint: run 7's `latin_ctx/best.pt` (epoch 20, val_acc 0.911438).** It wins run 6 on val_acc (+0.92 pt) and on total error count (−29), and the targeted `low_i` pathology holds at its run-6 floor (13 errors, no regression). Run 6's `latin_ctx-small/best.pt` (0.9023) is superseded. The training run itself used `char-dataset-ctx` (passed explicitly via the notebook's `DATASET_NAME`); the promoted default for *new* runs remains plain `char-dataset` — bumping the default to the uncapped variant is a separate, unmade decision.

### Next training items

- **~~Validate at full scale~~ — DONE (run 7).** The full uncapped `char-dataset-ctx` yields a modest +0.92 pt over the downsampled variant (see Run 7 conclusion 5). Worth using; not a step change.
- **~~Revisit `--epochs` budget~~ — DONE (run 7).** ~24-28 epochs is sufficient for this dataset/grid_mode combination; run 7's 36-epoch budget wasted 16 dead-tail epochs (see Run 7 conclusion 6). Do not run Latin on this config with a 36-epoch budget again.
- **~~Re-run `compare.py` end-to-end~~ — DONE (run 7), confirmatory only.** `compare.py --image detection/sample_craft_source.png` was run against the promoted run-7 `latin_ctx/best.pt` (staged into local `checkpoints/latin/`). On the 3 cleanly-segmented Latin regions the classifier agreed with standard OCR at 83/88/100% and handled `low_i`/`dig_1` correctly with no stroke-fragment collapse — consistent with the confused-pairs result. **This pass is a weak end-to-end signal**, not a rigorous A/B: (1) the synthetic fixture's text is kerned tighter than `char_classifier/segment.py`'s column-projection segmenter handles, so most apparent std-vs-clf disagreement in the longer regions is character-count/alignment drift, not classifier error; (2) the CJK regions route to the separate 1312-class `cjk` model (pre-existing ~0% collapse, unrelated to Latin run 7); (3) no run-6 `compare.py` baseline was ever captured, so there is nothing to diff against. A proper pipeline-level A/B would need a real Discord screenshot (real-context Latin text the segmenter splits correctly) run through both the run-6 and run-7 checkpoints side by side — worth doing only if a deployment decision comes to hinge on pipeline-level numbers. The rigorous signal (confused-pairs on the held-out test split, Run 7 conclusion 7) is already in hand and positive. `compare_preprocess.py` is unrelated here — it compares the six EasyOCR preprocessing variants, not the char_classifier, and does not exercise the run-7 checkpoint.

### Open decision — new-run default dataset

Run 7 confirmed the uncapped `char-dataset-ctx` beats the downsampled `char-dataset` default by +0.92 pt for Latin. **Deliberately not bumping the `train.py` / `remote_train.py` default to `char-dataset-ctx` yet**, because:

- The gain is marginal (+0.92 pt for ~2x data and ~2x epoch time).
- `char-dataset-ctx` only has a Latin split. kana/hangul/cjk migrate to real-context in runs 8+ (below), so the naming/default decision is made once, there, for all four scripts together.
- Changing the default touches the Drive/local dataset-naming divergence (see `remote_train.py` lines ~50-72), a standing footgun.

**Decision (2026-09-01, made as part of the runs-8+ migration below):** real-context data for **all four scripts** consolidates into `char-dataset/` (local) and a **refreshed `char-dataset.zip` on Drive that replaces the legacy isolated-tile zip in place** — a deliberate one-time break of the "frozen Drive zip names" rule (documented in `remote_train.py`'s `DATASET_NAME` comment), because a `char-dataset.zip` holding legacy data while local `char-dataset/` holds real-context data is exactly the footgun that rule was meant to prevent, and it worsens with three more scripts. Until that refreshed zip is uploaded, `char-dataset-ctx` stays the explicit `--dataset-name` for Latin runs (as run 7 did); once it is up, the notebooks/`train.py` default `char-dataset` means all-scripts real-context. Still open: whether `char-dataset/latin/` gets regenerated from the *uncapped* pipeline (to match run 7's `char-dataset-ctx`) or stays the run-6 downsampled set — not blocking the kana/hangul/cjk work.

### Runs 8+: kana/hangul/cjk real-context migration (started 2026-09-01)

The isolated-tile → real-context pivot that fixed Latin (runs 3-7) is being applied to kana/hangul/cjk. Status (2026-10-16): **kana (run 8), hangul (run 9) and cjk (run 10) are all trained, but all three used data with a generator bug that masked the target glyph in about 1/3 of kana/cjk tiles (see "Run 10: cjk" below).  The data has been regenerated; kana has been retrained (run 11, test top-1 0.977); hangul and cjk still need retraining.**

What was already in place: `render_chars_context.py` already dispatches `--scripts kana|hangul|cjk` (reusing `_build_kana`/`_build_hangul`/`_build_cjk` from `render_chars.py`); its render loop is script-agnostic. `train.py` / `remote_train.py` already accept those scripts and scope `checkpoints/<script>/` per script. The handoff's framing ("needs `render_chars_context.py` extended to those character sets") was stale.

The actual blocker was **font coverage**, fixed two ways:

1. **`.ttc`/`.otc` collection support.** `render_chars.py`'s `copy_system_fonts()` filtered for `.ttf`/`.otf` only, so Windows CJK fonts that ship *only* as `.ttc` (MS Gothic, Microsoft YaHei, YuGothic, MingLiU, MS JhengHei) were silently dropped — and `extract_cmap`'s `TTFont(path)` raises `TTLibFileIsCollectionError` on a `.ttc` without `fontNumber`, so any `.ttc` that did reach it returned an empty cmap and contributed nothing. This is a large part of why legacy CJK had only ~3-5 images/class. New shared helpers `iter_font_faces()` (yields each face index in a `.ttc`/`.otc`) and `build_font_meta()` (scans every face of every file → `(path, face_index, family, style, cmap)` 5-tuples, disambiguating `(family, style)` collisions by face index); `extract_cmap(path, font_number=)` and the `ImageFont.truetype(..., index=)` call sites thread the index through. `.ttf`/`.otf` still yield exactly one face, so Latin output is unchanged.
2. **Noto Sans CJK.** 28 OTFs (7 weights × jp/kr/sc/tc), added under `Models/Datasets/google-fonts/` (gitignored via the `Datasets/` pattern; generic name for future Google fonts), passed to the generators via `--extra-fonts-dir Models/Datasets/google-fonts`. Verified coverage: every one of the 4 language fonts covers 172/172 kana, 500/500 hangul, 3000/3000 CJK classes. This alone guarantees a dense per-class floor (28 fonts × 2 color modes × `variants_per_slot`) before any Windows fonts are added.

Smoke test (kana, `--variants-per-slot 1`, system fonts + Noto): 362 font files → **385 font faces with usable cmaps** (the extra faces are `.ttc` sub-faces previously skipped); **19,582 kana images / 172 classes ≈ 114/class**, vs legacy kana's ~12/class. Exit 0, `--update`-idempotent filenames intact.

Recommended per-script training config (few-shot regime — rougher loss surface, noisy tiny val sets): `--scheduler cosine-warm`, `--freeze-epochs 3`, ~30-40 epochs, `grid_mode` left at the `none` default (real string context makes `TileGrid3x3` redundant and, per run 6, actively harmful on every script). The five platform notebooks carry a "Runs 8+" history paragraph and per-script guidance in Cell 1.

#### Run 8 — kana (real-context, complete 2026-09-05)

> **Correction (2026-10-16):** this run trained and scored on data where about 32% of kana tiles were at least 60% pure black (the target glyph masked by a generator canvas bug; see "Run 10: cjk" below).  The numbers and failure modes in this section, including the small-kana collapse, are confounded by it and should be re-measured on the regenerated data.  **Re-measured in "Run 11: kana" below: test top-1 0.977 on clean data, and the small-kana collapse did not reappear.**

First non-Latin real-context run. Config: `--scripts kana --epochs 36 --freeze-epochs 3 --unfreeze-blocks 4 --batch-size 64 --backbone dinov2_vits14 --grid-mode none --mixup-alpha 0.2 --scheduler cosine-warm --lr 3e-4 --dataset-name char-dataset`. Colab T4, ~485-540 s/epoch (~5 hrs wall). Data: refreshed `char-dataset.zip` (real-context, Noto-backed), **172 classes** (all kana including the 3 obsolete `hira_3094/3095/3096` legacy never rendered), ~39,164 images, 70/15/15 split → 429 train batches/epoch. Checkpoint dir `checkpoints/kana/`.

**Results:**

| Metric | Value |
|---|---|
| best val_acc | **0.7115 @ epoch 36** (last epoch — still climbing, delta +0.0066) |
| val F1 macro @ best | 0.7308 (peak val F1 was 0.7486 @ epoch 26) |
| **test top-1 / top-3 / top-5** (`best.pt`) | **0.7180 / 0.7799 / 0.8054** |
| test macro precision / recall / f1 | 0.79 / 0.72 / 0.74 (support 5,875) |
| overfit_gap | ~ -0.30 (train_acc 0.41 vs val_acc 0.71 — no overfit; val > train from mixup + heavy aug) |

**cosine-warm restart effect:** first cosine cycle peaked at **0.7025 @ epoch 18**; the warm restart fired epoch 19 (LR 3.3e-6 → 3.0e-4), val_acc dipped to 0.668 @ epoch 20, then the second cycle climbed steadily to 0.7115 @ epoch 36. Net gain from the restart: **+0.9 pt** over the epoch-18 peak, matching the small post-restart gains seen in the Latin runs. val_acc was still rising at epoch 36 — a longer budget or a third restart might add a little more. For a repeat kana run, budget ~40-48 epochs (unlike Latin's run-7 finding of ~24-28; kana had not converged at 36).

**Failure modes (test-split, classes with f1 < 0.55):**
- **Small (sutegana) kana collapse** — the recurring pathology, analogous to Latin's `low_i` stroke-fragment issue. `kata_30a1` (small ア) f1 **0.18** (precision 0.10 — heavy over-prediction sink); `kata_30c3` (small ッ) 0.47; `kata_30e7`-family small ャ/ュ/ョ (`30e6` small ュ context 0.53, `30e8` small ョ 0.54); `kata_30ee` (small ヮ) 0.47; `hira_3045` (small ぇ) 0.33; `hira_3095/3096` (obsolete small kana) 0.49/low. Small kana differ from their full-size counterparts only in scale, which the target-glyph crop partially normalizes away.
- **Near-homoglyph hira/kata pairs** — top confused pairs: `kata_30f1` (ヱ) → `kata_30e6` (12), `hira_3045` → `hira_3046` (12), `hira_305b` (せ) → `kata_30b6` (ゼ) (10), `kata_30da`↔`hira_307a` (ペ/ぺ, 9+7), `kata_30d8`↔`hira_3078` (ヘ/へ, 9), `kata_30d9`↔`hira_3079` (ベ/べ, 7). The へ/ヘ hiragana-katakana pair is nearly identical by design; these are expected and not a regression signal.

No isolated-tile-style catastrophic collapse (no class at ~0% top-1). The floor is the small-kana f1≈0.2-0.5 band, not a zero.

#### Run 9 — hangul (real-context, complete 2026-09-09)

> **Correction (2026-10-16):** this run's data had about 5% of tiles at least 60% pure black and 34% at least 20% black (same generator canvas bug; see "Run 10: cjk" below).  The headline numbers and the complex-batchim failure analysis are confounded and should be re-measured on the regenerated data.

Config: `--scripts hangul --epochs 36 --freeze-epochs 3 --unfreeze-blocks 4 --batch-size 64 --backbone dinov2_vits14 --grid-mode none --mixup-alpha 0.2 --scheduler cosine-warm --lr 3e-4 --dataset-name char-dataset`. Kaggle T4, ~740-900 s/epoch (slower ~835-900 s in the second cosine cycle), ~8 h wall across two sessions. Data: real-context `char-dataset.zip` (Noto-backed), **500 classes** (top-500 hangul syllables), ~62,000 images, 70/15/15 split → ~680 train batches/epoch, 9,300 test images (~18.6/class). Checkpoint dir `checkpoints/hangul/`.

The run was interrupted at epoch 35/36 and re-run. The first restart archived the whole run instead of resuming: `train.py`'s `_check_and_archive_stale_run` treated the `git_commit` change from the intervening `e7a20ca` docs+infra commit as a training-plan mismatch (all 18 `_SIGNATURE_KEYS` matched exactly; `git_commit` was the sole diff). Fixed in `09938ca` — a bare commit-hash change no longer triggers the archive. The epoch-35 checkpoint was restored from `checkpoints/archive/hangul/20260909_run1_epoch35/` and resumed to completion.

**Results:**

| Metric | Value |
|---|---|
| best val_acc | **0.791935 @ epoch 35** (epoch 36 identical, delta 0.0 — converged/flat) |
| val F1 macro @ best epoch | 0.8023 (epoch 36: 0.8017) |
| **test top-1 / top-3 / top-5** (`best.pt`) | **0.8005 / 0.8896 / 0.9218** |
| test macro precision / recall / f1 | 0.86 / 0.80 / 0.81 (support 9,300) |
| overfit_gap | ~ -0.40 (train_acc 0.40 vs val_acc 0.79 — no overfit; val > train from mixup + heavy aug) |

Test top-1 (0.8005) landed **above** val_acc (0.7919) — the 15% test split is a shade easier than the 15% val split at 500 classes; not a concern.

**cosine-warm restart effect:** first cosine cycle peaked at **0.7828 @ epoch 13**, then flattened (0.778-0.783) as LR decayed to 3.3e-6 by epoch 18. The warm restart fired epoch 19 (LR 3.3e-6 → 3.0e-4), val_acc dipped one epoch to 0.7725, then the second cycle climbed to **0.7919 @ epoch 30** and held flat through 35-36. Net gain from the restart: **+0.91 pt** over the first-cycle peak — essentially identical to kana's +0.9 pt. **Unlike kana, hangul converged**: epochs 35 and 36 are bit-identical on val_acc. A repeat run needs ~36 epochs, not more; a third restart is unlikely to help.

**Failure modes (test-split):** the report is 500 rows of `syl_<hex codepoint>`; the low tail is a coherent pattern, not scattered noise.

- **Complex-batchim (final-consonant-cluster) syllables are the floor** — every class with f1 < 0.45 has a double final consonant (ㄳ/ㄵ/ㄶ/ㄺ/ㄻ/ㄼ/ㄽ/ㄾ/ㅄ) or a visually dense final: `syl_ac29` 갩 (ㄵ) f1 **0.22** (precision 0.13 — a heavy over-prediction sink, the `low_i` analogue), `syl_ac25` 갥 (ㄺ) 0.34, `syl_ac21` 갡 (ㄼ) 0.43, `syl_ac42` 걂 (ㄻ) 0.40, `syl_bbff` 믿 0.36. The target-glyph crop shrinks the whole syllable to a fixed box, so the batchim cluster — the part that disambiguates these — is rendered at just a few pixels and the model falls back on the shared initial+medial (the low tail is dominated by ㄱ-initial syllables with ㅐ/ㅔ/ㅕ medials plus a two-letter cluster).
- **Top confused pairs are all same-initial, same-or-adjacent-medial, differing only in the final consonant:** 겒→갡, 겐→걘, 걟→갥, 겑→갡 (all ×9-10), 걚→갡, 걒→갚, 갛→걓, 갂→갃 (×8). Two non-cluster pairs: 소→스 (ㅗ vs ㅡ medial, ×8) and 서→사 (ㅓ vs ㅏ, ×7) — short vertical-stroke vowels that differ by one tick.
- **High-precision / low-recall classes** (e.g. `syl_ac91` P 1.00 R 0.39, `syl_ac5f` P 0.86 R 0.35): the model rarely emits these but is right when it does — they lose their instances to the over-prediction sinks above.

No class at ~0% top-1 (min f1 0.22); no isolated-tile-style catastrophic collapse. The floor is the complex-batchim f1≈0.2-0.4 band. This is the direct hangul analogue of kana's small-kana collapse and Latin's `low_i`: a small, low-information sub-glyph feature that the fixed-box crop under-resolves.

#### Run 10: cjk (real-context, complete 2026-10-16; result capped by a generator bug)

Config: `--scripts cjk --epochs 36 --freeze-epochs 3 --unfreeze-blocks 4 --batch-size 64 --backbone dinov2_vits14 --grid-mode none --mixup-alpha 0.2 --scheduler cosine-warm --clip-grad 1.0 --lr 3e-4 --dataset-name char-dataset`.  Colab and Kaggle T4 across many sessions, 6,000 to 8,300 s/epoch (about 2 h).  Data: `char-dataset.zip` as of 2026-09-06, **3,000 classes**, 614,492 images, 70/15/15 split, 92,174 test images.  Checkpoint dir `checkpoints/cjk/`.

The run hit one crash mid-way: resuming from a checkpoint saved exactly at `freeze_epochs` was treated as already being in phase 2 (`resume_epoch >= freeze_epochs` at `train.py:355`).  Fixed in `7d3a74b` (`>=` to `>`); the run continued from there with no further crashes.

**Results (headline numbers, all capped by the data bug described below):**

| Metric | Value |
|---|---|
| best val_acc | **0.710244 @ epoch 12** (nothing after it was higher, 24 epochs since best at the end) |
| val_acc range, epochs 5 to 36 | 0.696 to 0.710 |
| **test top-1 / top-3 / top-5** (`best.pt`) | **0.7110 / 0.7410 / 0.7560** |
| test macro precision / recall / f1 | 0.89 / 0.71 / 0.77 (support 92,174) |

Epochs 1 to 3 (frozen backbone) sat at val_acc 0.016 to 0.021; unfreezing at epoch 4 jumped it to 0.682, and it then crept to 0.710 by epoch 12.  The cosine-warm restart fired at epoch 19 (LR 3.3e-6 back to 3.0e-4) and **gave nothing**: the second cycle stayed at 0.698 to 0.707 and f1 drifted down from about 0.765 to 0.722.  This is unlike kana and hangul, where the restart added about +0.9 pt.  train_acc alternated by epoch parity (about 0.37 on even epochs, 0.45 on odd epochs in the first cycle); not investigated.

**What the plateau actually was.**  The Cell 3 test report was print-only and had not been saved, so it was regenerated locally by running `best.pt` over the same deterministic test split (seed 42; top-1/3/5, macro P/R and support reproduced the Colab report exactly).  The failures were not tied to any font or mode:

| Slice | Accuracy |
|---|---|
| light / dark mode | 0.7109 / 0.7112 |
| variant v0 / v1 | 0.7124 / 0.7096 |
| 57 fonts, 10th to 90th percentile | 0.690 to 0.731 |

They were tied to how much of the tile was pure black (0,0,0):

| Pure-black share of the tile | Share of test set | Accuracy |
|---|---|---|
| under 40% | 66.9% | **0.999** |
| 60% or more | 33.1% | 0.13 |

0.669 x 0.999 + 0.331 x 0.13 is the headline 0.711.  99.9% of all misses had at least 5% pure black.  On every tile where the glyph was visible the model was essentially perfect; on the rest it guessed a handful of "sink" classes (U+4EEB predicted 1,431 times for 43 true samples, U+508C 1,371 for 30, U+5840 1,131 for 45; 81 classes with precision under 0.3; the 20 most-predicted classes absorb 40% of all misses).  Raw dataset PNGs, with no augmentation, show a solid black block covering most of the tile, with only fragments of a neighbouring character beside it.

**Root cause (generator, `render_chars_context.py`).**  The render canvas was a fixed 400 px square (`CANVAS_PAD * 2`) with the string drawn from its centre, so only 200 px lay to the right of the text origin.  At 64 pt a full-width kana/CJK glyph is about 64 px, so a target preceded by 3 neighbours (one draw in three) landed at x >= 392 and its crop window fell off the canvas; PIL fills an out-of-bounds crop with 0.  Reproduced by calling the real function on Noto Sans CJK SC: 32.5% of 600 tiles at least 60% black with the old canvas, 0.0% with a wider one.  This was present from the first real-context kana/hangul/cjk render (the dataset was generated on 2026-09-01) and went unnoticed because no per-slice accuracy or tile inspection was done.

**Other scripts, 3,000-image samples of the pre-fix `char-dataset` (Latin 1,240; measured 2026-10-16):**

| Script | Tiles >= 60% pure black | Tiles >= 20% |
|---|---|---|
| kana | 31.6% | 37.4% |
| cjk | 34.6% | 42.5% |
| hangul | 5.3% | 34.1% |
| Latin (`char-dataset` and `char-dataset-ctx`) | 0.2% | 8.8% / 8.7% |

Latin's narrower glyphs mostly keep the target on the canvas.  Its tiles at least 20% black fell from 8.8% to 6.7% after the fix; the remainder is real ink, since pure black is also the light-mode foreground colour.  (An earlier version of this note read the 46% of Latin tiles with at least 5% pure black as a black edge strip.  That was wrong: 43% of the regenerated, fixed Latin tiles are still above 5%, because heavy fonts in light mode legitimately exceed it.)  The effect on Latin accuracy is small and unmeasured.

**Consequences for earlier runs.**  Run 8 (kana, best val 0.7115) and run 9 (hangul, 0.7919) trained and scored on the same flawed data, so their headline numbers and their failure modes are confounded.  In particular the "small kana collapse" and the "complex-batchim floor", each with a sink class of precision around 0.1, look like the same sink-class behaviour seen here, and may be partly or wholly artifacts.  This is inference; it is not confirmed until those scripts are retrained on clean data.  The same applies to Latin runs 3 to 7 to a lesser, unmeasured degree.

**Fix and regeneration (2026-10-16).**
- `render_chars_context.py`: the canvas is now sized to the measured text (`text_w + 2*CANVAS_PAD` wide), and `render_context_tile` returns `None` rather than let PIL pad an out-of-bounds crop.  A post-render `_audit_black_tiles()` samples 600 tiles per script and prints a warning if more than 1% are at least 60% black.
- Smoke test on four cjk classes: 832 tiles, 0 mostly-black, same file count as the old dataset.
- Full regeneration result (all scripts, sampled 930 to 3,000 tiles each): **0.00% of tiles at least 60% black in every script**, audit clean on all four.  Class sets are identical to before and no class is empty.  Image counts: kana 39,164 (same), hangul 62,000 (same), cjk 614,492 (same), latin 87,032 (was 77,465; the extra Latin tiles come from the `.ttc`/`.otc` font faces added to `copy_system_fonts()` after the August Latin render).  Tiles at least 20% black: kana 37.4% to 0.0%, hangul 34.1% to 0.8%, cjk 42.5% to 0.2%, Latin 8.8% to 6.7% (real ink).
- Regeneration parameters were recovered from the old files (not recorded anywhere): `--variants-per-slot 2 --tile-size 64`, Noto Sans CJK via `--extra-fonts-dir ../../Datasets/google-fonts` for kana/hangul/cjk only, Latin without it.  Output went to a new directory (`char-dataset-fixed`), leaving the old dataset untouched until verified.
- The per-tile RNG seed includes the font file path string, so a re-render from a different location draws different neighbours and crops.  The regenerated set is a fresh random draw with a new train/val/test split, so its numbers are not directly comparable with runs 8 to 10.

**Restart safeguards.**  `train.py`'s run signature compared only hyperparameters and `dataset_name`, so regenerated data under the same name looked identical and a `--resume` would have loaded the old epoch-36 checkpoint.  The signature now includes the per-script content signatures from the dataset's `.dataset-manifest.json`; resuming against different data archives the old run (moved, not deleted) and starts fresh.  `print_report` now also writes `test_report.json` next to `progress.json`, and the five notebooks print it in their final results cell, so test results no longer exist only in cell output.

#### Run 11: kana (clean data, complete 2026-10-22)

Retrain of run 8 on the regenerated, black-tile-free `char-dataset`.  Config: `--scripts kana --epochs 36 --freeze-epochs 3 --unfreeze-blocks 4 --batch-size 64 --backbone dinov2_vits14 --grid-mode none --mixup-alpha 0.2 --scheduler cosine-warm --lr 3e-4 --dataset-name char-dataset`, identical to run 8.  Colab, about 440 s/epoch warm-up and 505 to 550 s/epoch fine-tune.  Started 2026-10-20; `progress.json` was last saved 2026-10-22 15:28 UTC.  `train.py` archived run 8 on start (`checkpoints/archive/kana/20261020_run1_epoch36/`, reason `resume_signature_mismatch`, only `dataset_signature` differing), which confirmed the new safeguard on a real session.  172 classes, 5,875 test images, new split (not split-comparable with run 8).

**Results:**

| Metric | Run 8 (black-tile bug) | Run 11 (clean) |
|---|---|---|
| best val_acc | 0.7115 @ 36 | **0.977872 @ epoch 32** |
| **test top-1 / top-3 / top-5** (`best.pt`) | 0.7180 / 0.7799 / 0.8054 | **0.9772 / 1.0000 / 1.0000** |
| test macro F1 | 0.74 | 0.977 |

The final epoch scored 0.9716 and `best.pt` is epoch 32, so epochs 29 to 36 sit in a 0.970 to 0.978 band.  First cosine cycle peaked at 0.9723 @ 16; the warm restart at epoch 19 dipped to 0.9607 and the second cycle added about +0.56 pt.  Val loss fell 1.20 to 0.91 and train accuracy stays about 0.45 to 0.50 (mixup plus heavy augmentation), so no overfitting.  Warm-up epochs 1 to 3 gave 0.148 / 0.186 / 0.229 and the first unfrozen epoch jumped to 0.9035.  The curve is flat, so more than 36 epochs is not expected to help kana.

**Failure modes (test split):** only 12 of 172 classes have f1 below 0.9 and the worst is 0.81 (`kata_30e7` ョ-family 0.81, `kata_30d9` 0.83, `hira_3078` 0.84, `kata_30d8` 0.85).  The run 8 small-kana collapse (f1 0.18 to 0.5 with a sink class at precision 0.10) is gone, which supports the reading that it was an artifact of the black tiles.  What remains are near-identical pairs: ヨ/ョ (`30e7`/`30e8`, 10), ベ/べ (9), へ/ヘ (7 and 4), あ/ぁ (6), ゆ/ュ-type small/full pairs, and the obsolete `hira_3095/3096` against か/け (5 each).  Hiragana/katakana look-alikes and small/full pairs differ little in shape, so this is plausibly a ceiling of the glyphs, not of training.  This is inferred from the confusion list, not tested.

### Remaining items after run 7

- **Generate + train kana/hangul/cjk real-context data.** Not yet done. `render_chars_context.py --scripts kana hangul cjk --extra-fonts-dir ../../Datasets/google-fonts` into `char-dataset/`, check per-class counts, zip the all-scripts `char-dataset/`, replace `char-dataset.zip` on Drive, then run the three trainings on Colab (`--scheduler cosine-warm`, ~30-40 epochs). Record results here.
- **Production integration is out of scope for the investigation.** Run 7's checkpoint proves the real-context approach; wiring the `Models/OCR/` CRAFT+char_classifier pipeline into the shipped `Translation/2-Image/` path (which currently uses EasyOCR) is a separate design question — replace vs post-correct EasyOCR, and how to handle the CJK routing gap — not a config flip.
