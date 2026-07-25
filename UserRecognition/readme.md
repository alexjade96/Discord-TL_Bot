# UserRecognition

Authorship attribution pipeline — given a chat message, rank which server members most likely sent it.

## Structure

```text
UserRecognition/
  identify.py                    ← inference API (public) + CLI
  tests/
    test_identify.py             ← unit tests for identify.py (mocked)
    test_collect_history.py      ← unit tests for collect_history.py (tmp_path)
  0-Data/
    training/
      collect_history.py         ← Discord message storage layer
      dataset.py                 ← build (text, label) train/val splits
      train.py                   ← TF-IDF + Logistic Regression classifier
      deploy.py                  ← dataset → train → install to ~/.tl-bot/authorship/
    testing/
      demo.py                    ← end-to-end pipeline demo (no bot required)
    data/
      guilds.jsonl               ← guild ID → name index (human-readable)
      {guild_id}/
        messages.jsonl           ← collected Discord messages
        users.jsonl              ← per-user stats
        channels.jsonl           ← per-channel scan metadata
        identity.jsonl           ← full member list snapshot (name → ID lookup)
        label_map.json           ← label int → username (written by dataset.py)
        train.jsonl / val.jsonl  ← train/val splits (written by dataset.py)
```

## Bot commands

| Command | Description |
| --- | --- |
| `/collect username [user2 ...]` | Collect message history for one or more users |
| `/collect --batch` | Collect all users listed in `data/{guild_id}/targets.txt` |
| `/identify <text>` | Rank likely authors of a message |

## Workflow

```bash
# 1. Collect message history via the bot
@TL-Bot /collect user1 user2 user3

# 2. Train (runs dataset.py then train.py in one shot)
.venv\Scripts\python.exe UserRecognition/0-Data/training/deploy.py --guild GUILD_ID

# 3. Verify
.venv\Scripts\python.exe UserRecognition/identify.py --guild GUILD_ID --text "some message"

# 4. Use in Discord
@TL-Bot /identify some message text here
```

## Model

TF-IDF (word 1-2grams + char 3-5grams) with Logistic Regression. Fast, CPU-only, effective at hundreds of messages per user. Model artifacts stored in `~/.tl-bot/authorship/{guild_id}/`. `identify.py` loads them at first call and caches at module level.

## Tests

```bash
pytest UserRecognition/tests/
```

All tests are mocked — no network, no real model files required.

## Model research

Deep-model experiments that aim to supersede the TF-IDF baseline live in
`Models/UserRecognition/` — see that directory's `README.md`. Nothing there is
imported by the bot; trained models reach production by installing to
`~/.tl-bot/authorship/{guild_id}/`, which `identify.py` already loads from.
