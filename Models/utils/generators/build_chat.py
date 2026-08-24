"""
build_chat.py -- generate chat-dataset/ from collected Discord message history.

Reads the production collection written by the bot's /collect command
(UserRecognition/0-Data/data/{guild_id}/) and emits a segmented training corpus
for the user recognition model in Models/UserRecognition/user_classifier/.

Run from Models/utils/generators/ (writes into ../../Datasets/):

    python build_chat.py --list
    python build_chat.py --guild 1502045408677986405
    python build_chat.py --guild 1502045408677986405 --chunk 4 --min-tokens 5
    python build_chat.py --guild 1502045408677986405 --holdout-channel general

Output layout:
    chat-dataset/
        {guild_id}/
            train.jsonl   val.jsonl   test.jsonl
            label_map.json
            meta.json

Differences from UserRecognition/0-Data/training/dataset.py (the TF-IDF baseline
builder), all of which matter for a model that can overfit:

  * Chronological split, not random.  The baseline shuffles before splitting, so
    adjacent messages from one conversation land in both train and val — topic
    and vocabulary leak across the split and inflate accuracy.  Here each
    author's messages are sorted by timestamp and cut oldest -> newest.
  * Three-way train/val/test.  The baseline emits only train/val, leaving no
    untouched set for a final number.
  * Bots excluded by default.  A bot's templated output is trivially separable
    and turns user recognition into bot-detection.
  * Message chunking, applied *before* the length filter.  Discord messages are
    short (median 4 tokens in the collected corpus); recognising a user from a
    single message is close to impossible.  --chunk N concatenates N consecutive messages from
    the same author into one sample, and --min-tokens then applies to the
    assembled chunk rather than to each message.  Filtering first would discard
    ~80% of a real corpus before those messages could contribute their style
    signal; short messages are weak alone but informative in aggregate.
  * Optional held-out channel.  --holdout-channel sends one channel entirely to
    test, which measures whether the model learned writing style or just the
    topic of the channel it trained on.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_HERE        = Path(__file__).parent                                   # Models/utils/generators/
_REPO_ROOT   = _HERE.parent.parent.parent
_SOURCE_ROOT = _REPO_ROOT / 'UserRecognition' / '0-Data' / 'data'
DATASET_DIR  = _REPO_ROOT / 'Models' / 'Datasets' / 'chat-dataset'

# Minimum distinct authors for the task to be user recognition at all.
_MIN_AUTHORS = 2


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_messages_by_user(guild_id: str, source_root: Path | None = None) -> dict[str, list[dict]]:
    """Return {user_id: [messages]} from data/{guild_id}/users/{user_id}.jsonl.

    The collection layer stores one file per user, so each file is already a
    clean per-class corpus and no grouping pass is needed here. A legacy
    messages.jsonl is still read if present, so an unmigrated guild still works.
    """
    root  = source_root or _SOURCE_ROOT
    gdir  = root / str(guild_id)
    users = gdir / 'users'

    by_user: dict[str, list[dict]] = {}
    if users.is_dir():
        for p in sorted(users.glob('*.jsonl')):
            rows = _read_jsonl(p)
            if rows:
                by_user[p.stem] = rows
        if by_user:
            return by_user

    # Fallback: pre-migration layout.
    for m in _read_jsonl(gdir / 'messages.jsonl'):
        uid = str(m.get('author_id') or '')
        if uid:
            by_user.setdefault(uid, []).append(m)
    return by_user


def load_identity(guild_id: str, source_root: Path | None = None) -> dict[str, dict]:
    """Return {user_id: identity_record}."""
    root = source_root or _SOURCE_ROOT
    return {r['user_id']: r for r in _read_jsonl(root / str(guild_id) / 'identity.jsonl')
            if 'user_id' in r}


def load_users(guild_id: str, source_root: Path | None = None) -> dict[str, str]:
    """Return {user_id: username}."""
    root = source_root or _SOURCE_ROOT
    return {r['user_id']: r.get('username', r['user_id'])
            for r in _read_jsonl(root / str(guild_id) / 'users.jsonl') if 'user_id' in r}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_messages(msgs: list[dict], chunk_size: int, joiner: str = ' ') -> list[dict]:
    """
    Group an author's chronologically-sorted messages into samples of chunk_size.

    chunk_size <= 1 returns one sample per message.  A trailing partial group is
    kept only when it holds at least half of chunk_size, so the tail of an
    author's history does not produce a stub sample far shorter than the rest.
    """
    msgs = sorted(msgs, key=lambda m: m.get('timestamp', ''))
    if chunk_size <= 1:
        return [
            {
                'text':       m['content_normalized'],
                'timestamp':  m.get('timestamp', ''),
                'channel_id': m.get('channel_id', ''),
                'n_messages': 1,
            }
            for m in msgs
        ]

    out: list[dict] = []
    for i in range(0, len(msgs), chunk_size):
        group = msgs[i:i + chunk_size]
        if len(group) < chunk_size and len(group) * 2 < chunk_size:
            break
        out.append({
            'text':       joiner.join(m['content_normalized'] for m in group),
            'timestamp':  group[0].get('timestamp', ''),
            'channel_id': group[0].get('channel_id', ''),
            'n_messages': len(group),
        })
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_chronological(samples: list[dict], val_frac: float, test_frac: float):
    """
    Split one author's samples oldest -> newest into (train, val, test).

    Chronological rather than random: a random split puts messages from the same
    conversation on both sides, so the model can score well by memorising topic
    instead of style.
    """
    samples = sorted(samples, key=lambda s: s.get('timestamp', ''))
    n = len(samples)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    # With few samples the fractions can round every holdout to zero; give val
    # and test one sample each before letting train take the rest.
    if n >= 3:
        n_val   = max(n_val, 1)
        n_test  = max(n_test, 1)
        n_train = max(n - n_val - n_test, 1)
        n_val   = min(n_val,  n - n_train)
        n_test  = n - n_train - n_val

    return (
        samples[:n_train],
        samples[n_train:n_train + n_val],
        samples[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(
    guild_id:         str,
    chunk:            int   = 1,
    min_tokens:       int   = 3,
    min_messages:     int   = 20,
    val_split:        float = 0.15,
    test_split:       float = 0.15,
    exclude_bots:     bool  = True,
    exclude_users:    list[str] | None = None,
    holdout_channel:  str | None = None,
    source_root:      Path | None = None,
    out_root:         Path | None = None,
) -> dict:
    """
    Build train/val/test splits for one guild.

    Returns a stats dict.  On failure the dict contains an 'error' key and
    nothing is written.
    """
    source_root = source_root or _SOURCE_ROOT
    out_root    = out_root    or DATASET_DIR

    raw_by_user = load_messages_by_user(guild_id, source_root)
    if not raw_by_user:
        return {'error': f'No messages found for guild {guild_id} under {source_root}.'}

    identity      = load_identity(guild_id, source_root)
    usernames     = load_users(guild_id, source_root)
    exclude_users = {str(u) for u in (exclude_users or [])}
    messages_in   = sum(len(v) for v in raw_by_user.values())

    # -- filter (user-level only; length is checked after chunking) ---------
    dropped = Counter()
    by_user: dict[str, list[dict]] = {}
    for uid, msgs in raw_by_user.items():
        if not uid:
            dropped['no_author'] += len(msgs)
            continue
        name = msgs[0].get('author_name', '') if msgs else ''
        if uid in exclude_users or name in exclude_users:
            dropped['excluded_user'] += len(msgs)
            continue
        if exclude_bots and identity.get(uid, {}).get('bot', False):
            dropped['bot'] += len(msgs)
            continue
        kept_msgs = [m for m in msgs if m.get('content_normalized', '').strip()]
        dropped['empty'] += len(msgs) - len(kept_msgs)
        if kept_msgs:
            by_user[uid] = kept_msgs

    kept    = [m for msgs in by_user.values() for m in msgs]
    too_few = {uid: len(v) for uid, v in by_user.items() if len(v) < min_messages}
    by_user = {uid: v for uid, v in by_user.items() if len(v) >= min_messages}

    # -- chunk, then apply the length floor to the assembled sample ---------
    by_user_samples: dict[str, list[dict]] = {}
    no_samples: list[str] = []
    for uid, msgs in by_user.items():
        samples  = chunk_messages(msgs, chunk)
        n_before = len(samples)
        samples  = [s for s in samples if len(s['text'].split()) >= min_tokens]
        dropped['short_chunks'] += n_before - len(samples)
        if samples:
            by_user_samples[uid] = samples
        else:
            no_samples.append(uid)

    if len(by_user_samples) < _MIN_AUTHORS:
        detail = (
            f'{len(by_user_samples)} author(s) survived filtering '
            f'(need >= {_MIN_AUTHORS}). '
            f'Dropped: {dict(dropped)}. '
            f'Below --min-messages {min_messages}: {too_few}.'
        )
        if no_samples:
            detail += (
                f' No chunk reached --min-tokens {min_tokens}: {no_samples} '
                f'(raise --chunk or lower --min-tokens).'
            )
        return {'error': f'Not enough users to build a recognition dataset. {detail}'}

    # -- label space -------------------------------------------------------
    sorted_uids = sorted(by_user_samples.keys())
    label_map   = {uid: idx for idx, uid in enumerate(sorted_uids)}

    # -- split -------------------------------------------------------------
    train_rows: list[dict] = []
    val_rows:   list[dict] = []
    test_rows:  list[dict] = []
    per_author: dict[str, dict] = {}

    for uid in sorted_uids:
        label    = label_map[uid]
        username = usernames.get(uid, identity.get(uid, {}).get('username', uid))
        samples  = by_user_samples[uid]

        if holdout_channel:
            held = [s for s in samples if s['channel_id'] == holdout_channel]
            rest = [s for s in samples if s['channel_id'] != holdout_channel]
            tr, va, _ = split_chronological(rest, val_split, test_split)
            te = held
        else:
            tr, va, te = split_chronological(samples, val_split, test_split)

        def _tag(rows, dest):
            for s in rows:
                dest.append({
                    'text':       s['text'],
                    'label':      label,
                    'author_id':  uid,
                    'username':   username,
                    'channel_id': s['channel_id'],
                    'timestamp':  s['timestamp'],
                    'n_messages': s['n_messages'],
                })

        _tag(tr, train_rows)
        _tag(va, val_rows)
        _tag(te, test_rows)

        per_author[uid] = {
            'username': username,
            'label':    label,
            'messages': len(by_user[uid]),
            'samples':  len(samples),
            'train':    len(tr),
            'val':      len(va),
            'test':     len(te),
        }

    # -- write -------------------------------------------------------------
    out_dir = Path(out_root) / str(guild_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(rows, name):
        with (out_dir / name).open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    _write(train_rows, 'train.jsonl')
    _write(val_rows,   'val.jsonl')
    _write(test_rows,  'test.jsonl')

    label_map_out = {
        uid: {'label': label_map[uid], 'username': per_author[uid]['username']}
        for uid in sorted_uids
    }
    (out_dir / 'label_map.json').write_text(
        json.dumps(label_map_out, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    warnings = []
    if not any(identity.get(u, {}).get('bot') is not None for u in identity):
        warnings.append(
            "identity.jsonl has no 'bot' field - re-run /collect to re-index members "
            "so bots can be excluded automatically. Use --exclude-user meanwhile."
        )
    total_train_tokens = sum(len(r['text'].split()) for r in train_rows)
    if train_rows and total_train_tokens / len(train_rows) < 10:
        warnings.append(
            f'Mean train sample length is {total_train_tokens / len(train_rows):.1f} tokens. '
            f'User recognition signal is weak below ~10; raise --chunk.'
        )
    if len(by_user) < 5:
        warnings.append(
            f'Only {len(by_user)} authors. Accuracy on a handful of classes is not '
            f'evidence the model generalises; collect more users before trusting it.'
        )

    meta = {
        'guild_id': str(guild_id),
        'config': {
            'chunk':           chunk,
            'min_tokens':      min_tokens,
            'min_messages':    min_messages,
            'val_split':       val_split,
            'test_split':      test_split,
            'exclude_bots':    exclude_bots,
            'exclude_users':   sorted(exclude_users),
            'holdout_channel': holdout_channel,
            'split_strategy':  'chronological',
        },
        'authors':      len(by_user),
        'train':        len(train_rows),
        'val':          len(val_rows),
        'test':         len(test_rows),
        'messages_in':  messages_in,
        'messages_kept': len(kept),
        'dropped':      dict(dropped),
        'per_author':   per_author,
        'warnings':     warnings,
    }
    (out_dir / 'meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    meta['out_dir'] = str(out_dir)
    return meta


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_guilds(source_root: Path | None = None) -> None:
    root = source_root or _SOURCE_ROOT
    if not root.exists():
        print(f'No collected data at {root}.')
        return

    names = {g['guild_id']: g.get('guild_name', g['guild_id'])
             for g in _read_jsonl(root / 'guilds.jsonl') if 'guild_id' in g}

    found = False
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        found = True
        by_user  = load_messages_by_user(d.name, root)
        identity = {r['user_id']: r for r in _read_jsonl(d / 'identity.jsonl') if 'user_id' in r}
        n_msgs   = sum(len(v) for v in by_user.values())
        bots     = sum(1 for uid in by_user if identity.get(uid, {}).get('bot'))
        built    = (DATASET_DIR / d.name).exists()
        print(f'  {d.name}  |  {names.get(d.name, d.name)}')
        print(f'      {n_msgs} messages  |  {len(by_user)} user(s)'
              f'{f" ({bots} bot)" if bots else ""}  |  '
              f'chat-dataset: {"built" if built else "not built"}')
        for uid, msgs in sorted(by_user.items(), key=lambda kv: -len(kv[1])):
            uname = identity.get(uid, {}).get('username') or (
                msgs[0].get('author_name', uid) if msgs else uid)
            tag = ' [bot]' if identity.get(uid, {}).get('bot') else ''
            print(f'        {uname:<24} {len(msgs):>6} messages{tag}')
    if not found:
        print(f'No guild directories under {root}.')


def main():
    p = argparse.ArgumentParser(
        description='Build chat-dataset/ from collected Discord message history.'
    )
    p.add_argument('--guild', default=None, help='Guild ID to build')
    p.add_argument('--list', action='store_true', help='List collected guilds and exit')
    p.add_argument('--chunk', type=int, default=1, metavar='N',
                   help='Concatenate N consecutive same-author messages per sample '
                        '(default 1). Raise this when messages are short.')
    p.add_argument('--min-tokens', type=int, default=3,
                   help='Drop messages with fewer than N tokens (default 3)')
    p.add_argument('--min-messages', type=int, default=20,
                   help='Drop authors with fewer than N surviving messages (default 20)')
    p.add_argument('--val-split', type=float, default=0.15,
                   help='Fraction of each author newest-but-one for val (default 0.15)')
    p.add_argument('--test-split', type=float, default=0.15,
                   help='Fraction of each author newest for test (default 0.15)')
    p.add_argument('--include-bots', action='store_true',
                   help='Keep bot authors (excluded by default — templated output '
                        'makes the task trivial)')
    p.add_argument('--exclude-user', nargs='+', default=[], metavar='ID_OR_NAME',
                   help='Additional user IDs or usernames to drop')
    p.add_argument('--holdout-channel', default=None, metavar='CHANNEL_ID',
                   help='Send this channel entirely to test, to measure style vs topic')
    args = p.parse_args()

    if args.list or not args.guild:
        list_guilds()
        if not args.guild:
            return

    stats = build(
        guild_id=args.guild,
        chunk=args.chunk,
        min_tokens=args.min_tokens,
        min_messages=args.min_messages,
        val_split=args.val_split,
        test_split=args.test_split,
        exclude_bots=not args.include_bots,
        exclude_users=args.exclude_user,
        holdout_channel=args.holdout_channel,
    )

    if 'error' in stats:
        print(f'Build failed: {stats["error"]}')
        raise SystemExit(1)

    print(f'Built chat-dataset for guild {args.guild}:')
    print(f'  Output:  {stats["out_dir"]}')
    print(f'  Authors: {stats["authors"]}')
    print(f'  Samples: {stats["train"]} train / {stats["val"]} val / {stats["test"]} test')
    print(f'  Kept {stats["messages_kept"]} of {stats["messages_in"]} messages '
          f'(dropped: {stats["dropped"] or "none"})')
    print('  Per author:')
    for info in stats['per_author'].values():
        print(f'    [{info["label"]}] {info["username"]:<20} '
              f'{info["messages"]:>5} msg -> {info["samples"]:>5} samples '
              f'({info["train"]}/{info["val"]}/{info["test"]})')
    for w in stats['warnings']:
        print(f'  WARNING: {w}')


if __name__ == '__main__':
    main()
