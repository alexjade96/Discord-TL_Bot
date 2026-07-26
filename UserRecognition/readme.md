# UserRecognition

User recognition pipeline — given a chat message, rank which server members most likely sent it.

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
      deploy.py                  ← dataset → train → install to ~/.tl-bot/user-recognition/
    testing/
      demo.py                    ← end-to-end pipeline demo (no bot required)
    data/
      guilds.jsonl               ← guild ID → name index (human-readable)
      {guild_id}/
        users/
          {user_id}.jsonl        ← that user's messages — one file per user
        users.jsonl              ← index: user_id → username, stats, in_guild
        channels.jsonl           ← per-channel scan metadata
        identity.jsonl           ← current member snapshot (overwritten each scan)
        authors.jsonl            ← every author ever seen posting (accumulates)
        label_map.json           ← label int → username (written by dataset.py)
        train.jsonl / val.jsonl  ← train/val splits (written by dataset.py)
```

### Per-user message storage

Messages are stored one file per user rather than in a single mixed
`messages.jsonl`. Each `users/{user_id}.jsonl` is already a clean per-class
corpus, so counting a user's samples, holding one user out, or dropping a user
from training is a file operation rather than a filter over the whole guild.
Filenames use the immutable `user_id`; `users.jsonl` is the index that maps it
to a username, so a username change never orphans a file.

`collect_history.py` exposes `list_user_ids()`, `load_user_messages()`,
`messages_by_user()`, and `user_message_counts()` for working per class, plus
`load_messages()` for the flat whole-guild view.

Migrating a guild collected under the old layout:

```bash
.venv\Scripts\python.exe UserRecognition/0-Data/training/collect_history.py --migrate
.venv\Scripts\python.exe UserRecognition/0-Data/training/collect_history.py --stats
```

Migration is idempotent — the legacy file is renamed to `messages.jsonl.migrated`
and re-running is a no-op. Both `dataset.py` and `build_chat.py` still read a
legacy `messages.jsonl` if no `users/` directory is present.

## Bot commands

| Command | Description |
| --- | --- |
| `/index` | Sweep channels and record everyone who has posted; stores no messages. Run once per guild before collecting users who have left |
| `/collect username [user2 ...]` | Collect message history for one or more users |
| `/collect "Display Name"` | Quote names containing spaces, or they parse as separate targets |
| `/collect <@id>` / `/collect 1234…` | Collect by mention or raw user ID |
| `/collect --batch` | Collect all users listed in `data/{guild_id}/targets.txt` |
| `/identify <text>` | Rank likely senders of a message |

`/index` and `/collect` share the `--channel`, `--since` and `--limit` flags.
`/index` requires **Manage Messages**.

### Collecting users who have left the server

Messages stay in channel history after their author leaves, and the scan filters
on a plain user ID rather than a `Member` object, so departed users are
collectable. Resolution falls through in this order:

1. `identity.jsonl` — current members, by username or display name
2. `authors.jsonl` — everyone ever seen posting, including departed users
3. a global user fetch, when an ID or @mention is given

`authors.jsonl` is built as a side effect of any scan: the loop already visits
every message, so it records each author it encounters, not just the target.
That means a departed user becomes resolvable **by name** once any collection
run has passed over a channel they posted in.

**Build the index first.** A scan only runs after a target resolves, so on a
guild that has never been swept, a departed user cannot resolve by name — the
index that would identify them is the thing the failed command would have
created. `/index` breaks that cycle: it sweeps every channel, records
every author, and stores no messages. Run it once, then collect by name:

```text
@TL-Bot /index
@TL-Bot /collect someonewholeft
```

A departed user who never posted in a scanned channel has no name to resolve
from — pass their **user ID or @mention**. There is no Discord API that maps a
username to an ID.

Results mark departed users, and their `users.jsonl` record carries
`in_guild: false`.

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

TF-IDF (word 1-2grams + char 3-5grams) with Logistic Regression. Fast, CPU-only, effective at hundreds of messages per user. Model artifacts stored in `~/.tl-bot/user-recognition/{guild_id}/`. `identify.py` loads them at first call and caches at module level.

## Tests

```bash
pytest UserRecognition/tests/
```

All tests are mocked — no network, no real model files required.

## Model research

`Models/UserRecognition/user_classifier/` holds the transformer model that
aims to supersede this TF-IDF baseline — see that directory's `README.md`.
Nothing there is imported by the bot.

`identify.py` serves both backends behind one API. Which one runs is decided by
the `model_type` field in `~/.tl-bot/user-recognition/{guild_id}/meta.json`:

| `model_type` | Backend | Installed by |
| --- | --- | --- |
| `tfidf` | TF-IDF + LogReg (this directory) | `0-Data/training/deploy.py` |
| `neural` | Fine-tuned transformer | `Models/UserRecognition/user_classifier/deploy.py` |

When both are installed the neural model wins; `deploy.py --remove <guild>`
falls back to TF-IDF. `identify(text, guild_id)` returns the same shape either
way, so `TL-Bot.py` is unaffected. `model_type(guild_id)` reports the active
backend.

Torch and transformers are imported only when a neural model is actually being
served — the TF-IDF path never touches them.
