"""Build (text, label) training pairs from collected per-user message files.

Usage:
    python dataset.py --guild GUILD_ID [--split 0.9] [--min-tokens 3] [--min-messages 20]
    python dataset.py --list
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_DATA_ROOT = Path(__file__).parent.parent / "data"


def _read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_messages_by_user(guild_id: str) -> dict[str, list[dict]]:
    """Return {user_id: [messages]} from users/{user_id}.jsonl.

    Falls back to a legacy messages.jsonl so an unmigrated guild still builds.
    """
    gdir  = _DATA_ROOT / str(guild_id)
    users = gdir / "users"
    by_user: dict[str, list[dict]] = {}
    if users.is_dir():
        for p in sorted(users.glob("*.jsonl")):
            rows = _read_jsonl(p)
            if rows:
                by_user[p.stem] = rows
        if by_user:
            return by_user
    for m in _read_jsonl(gdir / "messages.jsonl"):
        uid = m.get("author_id", "")
        if uid:
            by_user.setdefault(uid, []).append(m)
    return by_user


def _load_users(guild_id: str) -> dict:
    """Return {user_id: username} map."""
    p = _DATA_ROOT / str(guild_id) / "users.jsonl"
    out = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u = json.loads(line)
                out[u["user_id"]] = u.get("username", u["user_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def build_dataset(
    guild_id: str,
    split: float = 0.9,
    min_tokens: int = 3,
    min_messages: int = 20,
    seed: int = 42,
) -> dict:
    """
    Build stratified train/val splits from guild messages.

    Returns stats dict with keys: users, train, val, label_map.
    Writes train.jsonl and val.jsonl to the guild data directory.
    """
    users = _load_users(guild_id)

    # One file per user, so grouping is already done — just filter each.
    by_user = {
        uid: [m for m in msgs if m.get("token_count", 0) >= min_tokens]
        for uid, msgs in _load_messages_by_user(guild_id).items()
    }

    # Drop users with too few messages
    by_user = {uid: msgs for uid, msgs in by_user.items() if len(msgs) >= min_messages}

    if not by_user:
        return {
            "users": 0, "train": 0, "val": 0, "label_map": {},
            "error": f"No users with >= {min_messages} messages and >= {min_tokens} tokens after filtering.",
        }

    # Assign integer labels
    sorted_uids = sorted(by_user.keys())
    label_map = {uid: idx for idx, uid in enumerate(sorted_uids)}

    # Stratified split per user
    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows:   list[dict] = []

    for uid, msgs in by_user.items():
        rng.shuffle(msgs)
        cut = max(1, int(len(msgs) * split))
        label = label_map[uid]
        username = users.get(uid, uid)
        for m in msgs[:cut]:
            train_rows.append({
                "text":     m["content_normalized"],
                "label":    label,
                "author_id": uid,
                "username": username,
            })
        for m in msgs[cut:]:
            val_rows.append({
                "text":     m["content_normalized"],
                "label":    label,
                "author_id": uid,
                "username": username,
            })

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    out_dir = _DATA_ROOT / str(guild_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(rows, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    _write(train_rows, "train.jsonl")
    _write(val_rows,   "val.jsonl")

    # Write label map
    label_map_out = {uid: {"label": idx, "username": users.get(uid, uid)} for uid, idx in label_map.items()}
    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map_out, f, indent=2, ensure_ascii=False)

    return {
        "users":     len(by_user),
        "train":     len(train_rows),
        "val":       len(val_rows),
        "label_map": label_map_out,
    }


def list_guilds() -> None:
    if not _DATA_ROOT.exists():
        print("No data directory found.")
        return

    guilds_file = _DATA_ROOT / "guilds.jsonl"
    guild_names: dict[str, str] = {}
    if guilds_file.exists():
        with guilds_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                    guild_names[g["guild_id"]] = g.get("guild_name", g["guild_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    for d in sorted(_DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        gid = d.name
        name = guild_names.get(gid, gid)
        by_user   = _load_messages_by_user(gid)
        msg_count = sum(len(v) for v in by_user.values())
        usernames = _load_users(gid)
        print(f"  {gid}  |  {name}  |  {msg_count} messages  |  {len(by_user)} user(s)")
        for uid, msgs in sorted(by_user.items(), key=lambda kv: -len(kv[1])):
            uname = usernames.get(uid) or (msgs[0].get("author_name", uid) if msgs else uid)
            print(f"      {uname:<24} {len(msgs):>6} messages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build user recognition dataset from collected messages.")
    sub = parser.add_subparsers(dest="cmd")

    build_p = sub.add_parser("build", help="Build train/val splits for a guild")
    build_p.add_argument("--guild", required=True, help="Guild ID")
    build_p.add_argument("--split", type=float, default=0.9, help="Train fraction (default 0.9)")
    build_p.add_argument("--min-tokens", type=int, default=3, help="Min token count per message (default 3)")
    build_p.add_argument("--min-messages", type=int, default=20, help="Min messages per user (default 20)")
    build_p.add_argument("--seed", type=int, default=42, help="Random seed")

    sub.add_parser("list", help="List all guilds with collected data")

    # Support flat arg style: dataset.py --guild X ...
    parser.add_argument("--guild", help="Guild ID (shorthand for build subcommand)")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--min-messages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    if args.cmd == "list" or getattr(args, "list", False):
        list_guilds()
    elif args.cmd == "build" or getattr(args, "guild", None):
        guild_id = args.guild if args.cmd == "build" else args.guild
        stats = build_dataset(
            guild_id=guild_id,
            split=args.split,
            min_tokens=args.min_tokens,
            min_messages=args.min_messages,
            seed=args.seed,
        )
        if "error" in stats:
            print(f"Dataset build failed: {stats['error']}")
        else:
            print(f"Built dataset for guild {guild_id}:")
            print(f"  Users:  {stats['users']}")
            print(f"  Train:  {stats['train']} samples")
            print(f"  Val:    {stats['val']} samples")
            print("  Labels:")
            for uid, info in stats["label_map"].items():
                print(f"    [{info['label']}] {info['username']} ({uid})")
    else:
        parser.print_help()
