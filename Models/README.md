# Abstract

Package for improving text recognition/OCR models.  The general idea is to generate a set of images of script-dependent characters in a 3 by 3 grid, placing characters adjacent to each other to fill in 9 boxes:

| X | X | X |
|:---:|:---:|:---:|
| X | X | X |
| X | X | X |

Increase the dataset of these by transforming the characters in these images by rotating, shifting, etc. but ensuring the characters themselves would line up to a human reader.  Further expansion can include non-same characters next to the focused character:

| Y | Y | Y |
|:---:|:---:|:---:|
| Y | **X** | Y |
| Y | Y | Y |

---

## Grid Configurations

The 3×3 same-character grid and the focal+neighbour grid are the two base cases.
Additional configurations expand coverage:

**Diagonal isolation** — neighbours only on diagonal corners, leaving cardinal
cells empty.  Tests whether the model relies on horizontal/vertical adjacency or
can generalise to corner-only context:

| Y |   | Y |
|:---:|:---:|:---:|
|   | **X** |   |
| Y |   | Y |

**Row context** — a single row of the same character, with the focal character
centred.  Simulates a real line of text where the character repeats (e.g. a row
of ideographs):

|   |   |   |
|:---:|:---:|:---:|
| Y | **X** | Y |
|   |   |   |

**Mixed script border** — neighbours drawn from a different script than the focal
character.  Simulates real-world multilingual text: a Latin character surrounded
by CJK, or a Hangul syllable bordered by ASCII punctuation:

| A | A | A |
|:---:|:---:|:---:|
| A | **가** | A |
| A | A | A |

**Density gradient** — the 3×3 grid can be extended to 5×5 or 7×7 with the
focal character remaining centred.  Larger grids produce more context signal at
the cost of a smaller focal character relative to the image canvas.

---

## Augmentation Strategy

Augmentations are applied to the whole grid image after composition so that
spatial relationships between cells are preserved.  Each transform should keep
the grid legible — a human reader should still be able to identify which cell is
which.

**Geometric** (order-independent, can be combined):
- Rotation: ±5° to ±15° in small increments; avoid multiples of 90° unless testing
  rotational invariance specifically
- Translation: shift the entire grid up to ~10% of canvas width/height
- Scale: uniform zoom ±15%; preserves aspect ratio so character stroke widths
  remain proportional
- Shear: horizontal or vertical skew up to ~10° to simulate camera angle

**Photometric**:
- Brightness and contrast variation to simulate screenshots at different display
  calibrations
- JPEG compression at varying quality levels to simulate real-world image
  transcoding artefacts
- Gaussian noise and blur to match real-world image degradation
- Dark-mode inversion: white characters on a dark background to cover
  light-on-dark rendering environments

**Typography-level** (applied before grid composition):
- Font weight variation across the same character: render X in Regular, Bold, and
  Light side by side as separate grid variants
- Sub-pixel anti-aliasing differences across rendering engines
- Stroke width perturbation for synthetic bold/thin variants

Each base grid image should produce a minimum of N augmented copies to ensure the
model does not overfit to a single rendering.  The exact N is determined by the
rarity of the character and the size of the existing dataset for that script.

---

## Character Selection

Not all codepoints within a Unicode block are equally useful as training targets.
Character selection should be weighted by:

- **Frequency in real-world text** — high-frequency characters (common Hangul
  syllables, frequent CJK ideographs, standard Latin letters) are more valuable
  than rare codepoints that almost never appear in practice
- **Visual similarity clusters** — characters that are commonly confused by OCR
  (e.g. 己 vs 已 vs 巳, or l vs I vs 1) should be over-represented so the model
  learns the distinguishing strokes
- **Script mixing frequency** — for the mixed-border grids, neighbour characters
  should be drawn from scripts that actually co-occur with the focal script in
  real messages, not arbitrary Unicode blocks

---

## Neighbour Character Selection

The choice of Y in the focal+neighbour grid matters more than it may appear.
Three neighbour selection strategies, each producing a distinct training signal:

1. **Same script, different character** — Y is drawn from the same Unicode block
   as X.  Teaches inter-character spacing norms within a script.

2. **Cross-script** — Y is drawn from a different script.  The most relevant
   case for this project: Hangul/CJK surrounded by Latin, or Latin surrounded
   by Arabic.

3. **Adversarial** — Y is a character with high visual similarity to X (a
   near-homoglyph or a character that shares dominant strokes).  The hardest
   case; forces the model to rely on fine-grained stroke differences rather
   than coarse shape.

Combining all three strategies for each focal character X produces a richer
dataset than any single strategy alone.

---

## Scaling to Larger Grids

As grids grow beyond 3×3, the definition of "focal character" and "context
character" generalises:

- In a 5×5 grid the focal cell is still the centre; the 16 surrounding cells
  form two rings of context (immediate neighbours vs. outer ring), which can
  be filled independently with different Y characters to encode two levels of
  context distance:

| Z | Z | Z | Z | Z |
|:---:|:---:|:---:|:---:|:---:|
| Z | Y | Y | Y | Z |
| Z | Y | **X** | Y | Z |
| Z | Y | Y | Y | Z |
| Z | Z | Z | Z | Z |

- In a 7×7 grid the outer ring is far enough from centre that it approximates
  the visual context a model sees when sliding a detection window across a
  dense page of text.

The label for each grid image remains the identity of the focal character only;
context cells are unlabelled noise from the model's perspective, which is
exactly the condition the model faces in real OCR.

---

## Integration with the OCR Training Pipeline

The grids produced here feed directly into the character-recognition fine-tuning
stage.  The intended flow:

1. Generate base grids for each target character and script combination
2. Apply augmentation to reach the target count per character
3. Store as labelled image crops (label = Unicode codepoint of focal character X)
4. Combine with real image crops from the existing data collection pipeline to
   produce the final training split
5. Fine-tune the recognition model on the combined dataset
6. Evaluate on a held-out set of real screenshots, not synthetic grids, to
   confirm that synthetic data generalises

The synthetic grids are intended to supplement — not replace — real data.  The
primary value is filling coverage gaps for rare characters or scripts that appear
infrequently in collected data but are important to handle correctly.
