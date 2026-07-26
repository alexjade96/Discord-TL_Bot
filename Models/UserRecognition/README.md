# Models/UserRecognition — user recognition model research

Deep-model user recognition. Nothing here is imported by `TL-Bot.py`; it
reaches production only through `deploy.py`.

## Relationship to `/UserRecognition`

The feature is split the same way OCR is, between `Translation/2-Image/`
(production) and `Models/OCR/` (research):

| | Location | Contents |
| --- | --- | --- |
| **Production** | `UserRecognition/` | `identify.py` inference API, `0-Data/training/` TF-IDF + LogReg baseline, collected message data |
| **Research** | `Models/UserRecognition/` (here) | `user_classifier/` transformer model that supersedes the baseline |

Both backends install to `~/.tl-bot/user-recognition/{guild_id}/`. `identify.py`
dispatches on the `model_type` field in `meta.json`, so deploying from here
takes over from the baseline with no change to the bot. Removing it falls back.

## Layout

```
Models/UserRecognition/
  user_classifier/
    data.py           ← ChatDataset, get_dataloaders, style-preserving augments
    model_builder.py  ← TransformerClassifier + create_model / create_tokenizer
    engine.py         ← train_step / eval_step / train_loop
    model_utils.py    ← param counts, unfreeze_encoder
    utils.py          ← device, seed, checkpoint save/load
    stats.py          ← plot_curves, print_report (incl. chance baseline)
    train.py          ← training CLI, two-phase, checkpoints/<guild_id>/
    predict.py        ← single-text inference CLI
    deploy.py         ← install to ~/.tl-bot/user-recognition/<guild_id>/
  checkpoints/<guild_id>/
    best.pt  last.pt  config.json  class_names.json  progress.json  curves.png
  tests/
    test_build_chat.py         ← dataset builder (split integrity, chunking, filtering)
    test_user_classifier.py  ← data loading, augments, deploy/remove
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
.\..\..\.venv\Scripts\python.exe -m user_classifier.train --guild GUILD_ID
.\..\..\.venv\Scripts\python.exe -m user_classifier.train --guild GUILD_ID \
    --backbone distilbert-base-multilingual-cased --epochs 20

# 4. Inspect before deploying
.\..\..\.venv\Scripts\python.exe -m user_classifier.predict --guild GUILD_ID --text "..."

# 5. Deploy — identify.py picks it up automatically
.\..\..\.venv\Scripts\python.exe -m user_classifier.deploy --guild GUILD_ID
.\..\..\.venv\Scripts\python.exe -m user_classifier.deploy --list
.\..\..\.venv\Scripts\python.exe -m user_classifier.deploy --remove GUILD_ID   # revert to TF-IDF
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

**Checkpoint selection** — `--select-metric` decides which epoch becomes
`best.pt`, defaulting to **`f1`** (macro-F1) rather than accuracy. Chat corpora
are heavily imbalanced: in the first real guild one user held 53% of samples, so
selecting on accuracy picked an epoch that predicted the majority user well and
everyone else badly. On that data the two criteria disagreed sharply —

| Selection | Epoch chosen | Val acc | Val F1 | **Test macro-F1** |
| --- | --- | --- | --- | --- |
| `val_acc` | 10 | 0.9200 | 0.6830 | 0.4279 |
| `f1` (default) | 15 | 0.9067 | 0.8909 | **0.5238** |

`train.py` reloads `best.pt` before the final test report, so the printed
metrics describe the checkpoint `deploy.py` installs. Reporting the live
final-epoch model instead — as it previously did — described a model that was
never saved.

No mixup: interpolating token embeddings between two authors produces text
belonging to neither. Label smoothing covers the same regularisation need.

## Data requirements

**Training state: no model trained yet.** The TF-IDF baseline is what the bot
serves.

Two guilds collected as of 2026-07-25:

| Guild | Messages | Authors after filtering | Samples at `--chunk 4` |
| --- | --- | --- | --- |
| `1502045408677986405` | 312 | 1 (build fails) | — |
| `122861720501878784` | 4,054 | 3 | 683 / 145 / 145 |

The first guild is unusable: 2 authors, one of them TL-Bot, in one channel, and
the human's messages are mostly bot commands (`$hello`, `/translate`). A model
trained on it learns "bot reply vs slash command", not user recognition.

The second is genuinely usable but thin — 3 authors, and heavily imbalanced
(2,158 / 1,503 / 227 messages). Two more members were excluded by
`--min-messages 20`: ringoshiiro9468 had 3 collected messages, and Groovy is a
bot.

Rough threshold before results mean anything: **5+ human authors, 500+ messages
each at ≥10 tokens, across ≥2 channels.** `build_chat.py` refuses to build below
2 authors and emits warnings above that; `train.py` echoes those warnings at
startup. The nearest win is collecting the remaining members of the second
guild — `/collect` now handles departed users and quoted display names, which
is what blocked the last few attempts.

## Tests

```bash
pytest Models/UserRecognition/tests/
```

Mocked — no network, no backbone download, no trained model required.
