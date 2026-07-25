# Models/UserRecognition — authorship model research

Deep-model authorship attribution. Nothing here is imported by `TL-Bot.py`; it
reaches production only through `deploy.py`.

## Relationship to `/UserRecognition`

The feature is split the same way OCR is, between `Translation/2-Image/`
(production) and `Models/OCR/` (research):

| | Location | Contents |
| --- | --- | --- |
| **Production** | `UserRecognition/` | `identify.py` inference API, `0-Data/training/` TF-IDF + LogReg baseline, collected message data |
| **Research** | `Models/UserRecognition/` (here) | `author_classifier/` transformer model that supersedes the baseline |

Both backends install to `~/.tl-bot/authorship/{guild_id}/`. `identify.py`
dispatches on the `model_type` field in `meta.json`, so deploying from here
takes over from the baseline with no change to the bot. Removing it falls back.

## Layout

```
Models/UserRecognition/
  author_classifier/
    data.py           ← ChatDataset, get_dataloaders, style-preserving augments
    model_builder.py  ← TransformerClassifier + create_model / create_tokenizer
    engine.py         ← train_step / eval_step / train_loop
    model_utils.py    ← param counts, unfreeze_encoder
    utils.py          ← device, seed, checkpoint save/load
    stats.py          ← plot_curves, print_report (incl. chance baseline)
    train.py          ← training CLI, two-phase, checkpoints/<guild_id>/
    predict.py        ← single-text inference CLI
    deploy.py         ← install to ~/.tl-bot/authorship/<guild_id>/
  checkpoints/<guild_id>/
    best.pt  last.pt  config.json  class_names.json  progress.json  curves.png
  tests/
    test_build_chat.py         ← dataset builder (split integrity, chunking, filtering)
    test_author_classifier.py  ← data loading, augments, deploy/remove
```

## Dataset

Training data comes from `Models/Datasets/chat-dataset/{guild_id}/`, built by
`Models/Datasets/build_chat.py` from the bot's collected message history. See
the "Chat dataset" section of `Models/PIPELINE.md` for what that builder does
differently from the baseline's `dataset.py` — chronological splitting, bot
exclusion, message chunking, and a held-out channel option.

## Workflow

```powershell
# 1. Collect message history via the bot
#    @TL-Bot /collect user1 user2 user3

# 2. Build the segmented corpus (from Models/Datasets/)
.\..\..\.venv\Scripts\python.exe build_chat.py --list
.\..\..\.venv\Scripts\python.exe build_chat.py --guild GUILD_ID --chunk 4

# 3. Train (from Models/UserRecognition/)
.\..\..\.venv\Scripts\python.exe -m author_classifier.train --guild GUILD_ID
.\..\..\.venv\Scripts\python.exe -m author_classifier.train --guild GUILD_ID \
    --backbone distilbert-base-multilingual-cased --epochs 20

# 4. Inspect before deploying
.\..\..\.venv\Scripts\python.exe -m author_classifier.predict --guild GUILD_ID --text "..."

# 5. Deploy — identify.py picks it up automatically
.\..\..\.venv\Scripts\python.exe -m author_classifier.deploy --guild GUILD_ID
.\..\..\.venv\Scripts\python.exe -m author_classifier.deploy --list
.\..\..\.venv\Scripts\python.exe -m author_classifier.deploy --remove GUILD_ID   # revert to TF-IDF
```

## Model

`TransformerClassifier` — a pretrained multilingual encoder, mean-pooled over
the attention mask, into the same two-layer MLP head `char_classifier` uses.
Mean pooling rather than CLS because phase 1 trains with the encoder frozen, and
an un-fine-tuned CLS token carries no sentence representation.

Backbones (`--backbone`):

| Name | Params | Notes |
| --- | --- | --- |
| `xlm-roberta-base` | 278M | Default. Multilingual — this is a translation bot's server, so chat is expected to be mixed-script |
| `distilbert-base-multilingual-cased` | 134M | ~2× faster on CPU, lower ceiling |

Two-phase training mirrors `char_classifier`: frozen-encoder head warm-up, then
fine-tune the top `--unfreeze-layers` with a differential LR. The encoder LR
multiplier is 0.05 rather than the image pipeline's 0.1, because a pretrained
language encoder drifts fast on a few hundred short samples.

No mixup: interpolating token embeddings between two authors produces text
belonging to neither. Label smoothing covers the same regularisation need.

## Data requirements

**Training state: no model trained yet.** The TF-IDF baseline is what the bot
serves.

The collected corpus is not yet sufficient to train this model. As of
2026-07-25 the only collected guild holds 312 messages from 2 authors, one of
which is TL-Bot itself, in a single channel, with a median of 4 tokens per
message — and the human's messages are mostly bot commands (`$hello`,
`/translate`). A model trained on that learns "bot reply vs slash command", not
authorship.

Rough threshold before results mean anything: **5+ human authors, 500+ messages
each at ≥10 tokens, across ≥2 channels.** `build_chat.py` refuses to build below
2 authors and emits warnings above that; `train.py` echoes those warnings at
startup.

## Tests

```bash
pytest Models/UserRecognition/tests/
```

Mocked — no network, no backbone download, no trained model required.
