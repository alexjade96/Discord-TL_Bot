from __future__ import annotations

import json
import re
from pathlib import Path

_DATA_ROOT = Path(__file__).parent.parent / "data"

_MENTION_RE     = re.compile(r'<@!?\d+>')
_CHANNEL_RE     = re.compile(r'<#\d+>')
_ROLE_RE        = re.compile(r'<@&\d+>')
_URL_RE         = re.compile(r'https?://\S+')
_EMOJI_RE       = re.compile(r'<a?:\w+:\d+>')


def normalize_content(content: str) -> str:
    content = _MENTION_RE.sub('[USER]', content)
    content = _CHANNEL_RE.sub('[CHANNEL]', content)
    content = _ROLE_RE.sub('[ROLE]', content)
    content = _URL_RE.sub('[URL]', content)
    content = _EMOJI_RE.sub('[EMOJI]', content)
    return content.strip()


def _guild_dir(guild_id: str) -> Path:
    d = _DATA_ROOT / str(guild_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _users_dir(guild_id: str) -> Path:
    """Per-user message store: data/{guild_id}/users/{user_id}.jsonl.

    One file per user rather than a single mixed messages.jsonl. Each file is
    already a clean per-class corpus, so building training splits, counting a
    user's samples, or dropping one user from the dataset is a file operation
    rather than a filter over the whole guild. users.jsonl is the index mapping
    each user_id to its username.
    """
    d = _guild_dir(guild_id) / "users"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_message_path(guild_id: str, user_id: str) -> Path:
    return _users_dir(guild_id) / f"{user_id}.jsonl"


def list_user_ids(guild_id: str) -> list[str]:
    """Return every user_id that has a message file for this guild."""
    d = _guild_dir(guild_id) / "users"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


def load_user_messages(guild_id: str, user_id: str) -> list[dict]:
    """Return all stored messages for one user."""
    p = user_message_path(guild_id, user_id)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_seen_ids(guild_id: str) -> set:
    """Return set of message_ids already stored for this guild, across all users."""
    seen = set()
    for uid in list_user_ids(guild_id):
        for rec in load_user_messages(guild_id, uid):
            mid = rec.get("message_id")
            if mid:
                seen.add(mid)
    return seen


def save_message(record: dict, guild_id: str, seen_ids: set) -> bool:
    """Append record to that author's users/{user_id}.jsonl. Returns True if new."""
    mid = record["message_id"]
    if mid in seen_ids:
        return False
    uid = str(record.get("author_id") or "unknown")
    with user_message_path(guild_id, uid).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    seen_ids.add(mid)
    return True


def save_user(record: dict, guild_id: str) -> None:
    """Upsert user record by user_id. Keeps earliest first_message_at across runs."""
    p = _guild_dir(guild_id) / "users.jsonl"
    users = {}
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        u = json.loads(line)
                        users[u["user_id"]] = u
                    except (KeyError, json.JSONDecodeError):
                        pass
    uid = record["user_id"]
    if uid in users:
        existing = users[uid]
        old_first = existing.get("first_message_at")
        existing.update(record)
        # Restore the earlier timestamp if the pre-existing one came first
        new_first = record.get("first_message_at")
        if old_first and new_first and old_first < new_first:
            existing["first_message_at"] = old_first
        users[uid] = existing
    else:
        users[uid] = record
    with p.open("w", encoding="utf-8") as f:
        for u in users.values():
            f.write(json.dumps(u, ensure_ascii=False) + "\n")


def save_channel(record: dict, guild_id: str) -> None:
    """Upsert channel record by channel_id. Accumulates message_count_collected."""
    p = _guild_dir(guild_id) / "channels.jsonl"
    channels = {}
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        c = json.loads(line)
                        channels[c["channel_id"]] = c
                    except (KeyError, json.JSONDecodeError):
                        pass
    cid = record["channel_id"]
    if cid in channels:
        prior = channels[cid].get("message_count_collected", 0)
        channels[cid] = record
        channels[cid]["message_count_collected"] = prior + record.get("message_count_collected", 0)
    else:
        channels[cid] = record
    with p.open("w", encoding="utf-8") as f:
        for c in channels.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def save_guild(guild_id: str, guild_name: str, indexed_at: str) -> None:
    """Upsert guild entry in data/guilds.jsonl for human-readable guild ID → name mapping."""
    p = _DATA_ROOT / "guilds.jsonl"
    guilds: dict = {}
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        g = json.loads(line)
                        guilds[g["guild_id"]] = g
                    except (KeyError, json.JSONDecodeError):
                        pass
    guilds[guild_id] = {"guild_id": guild_id, "guild_name": guild_name, "indexed_at": indexed_at}
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for g in guilds.values():
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def save_identity(records: list, guild_id: str) -> None:
    """Overwrite identity.jsonl with the full current member list for this guild."""
    p = _guild_dir(guild_id) / "identity.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_identity(guild_id: str) -> dict:
    """Load identity table.

    Returns {"by_id": {user_id: record}, "by_name": {lower_name: record}}.
    by_name covers both username and display_name (display_name wins on collision).
    """
    p = _guild_dir(guild_id) / "identity.jsonl"
    by_id: dict = {}
    by_name: dict = {}
    if not p.exists():
        return {"by_id": by_id, "by_name": by_name}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                uid = r["user_id"]
                by_id[uid] = r
                by_name[r["username"].lower()] = r
                dn = r.get("display_name", "")
                if dn and dn.lower() != r["username"].lower():
                    by_name[dn.lower()] = r
            except (KeyError, json.JSONDecodeError):
                pass
    return {"by_id": by_id, "by_name": by_name}


def save_authors(records: list, guild_id: str) -> int:
    """Upsert observed message authors into authors.jsonl. Returns the number of new entries.

    Unlike identity.jsonl (a snapshot of *current* members, overwritten on every
    scan), this accumulates every author ever seen in channel history — including
    users who have since left the server. It is what makes a departed user
    resolvable by name.
    """
    p = _guild_dir(guild_id) / "authors.jsonl"
    authors: dict = {}
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        a = json.loads(line)
                        authors[a["user_id"]] = a
                    except (KeyError, json.JSONDecodeError):
                        pass

    new_count = 0
    for r in records:
        uid = r.get("user_id")
        if not uid:
            continue
        if uid in authors:
            # Keep the earliest first_seen_at; refresh everything else.
            old_first = authors[uid].get("first_seen_at")
            authors[uid].update(r)
            new_first = r.get("first_seen_at")
            if old_first and new_first and old_first < new_first:
                authors[uid]["first_seen_at"] = old_first
        else:
            authors[uid] = r
            new_count += 1

    with p.open("w", encoding="utf-8") as f:
        for a in authors.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return new_count


def load_authors(guild_id: str) -> dict:
    """Load the observed-author index.

    Returns {"by_id": {user_id: record}, "by_name": {lower_name: record}}, the
    same shape as load_identity so callers can fall through from one to the other.
    """
    p = _guild_dir(guild_id) / "authors.jsonl"
    by_id: dict = {}
    by_name: dict = {}
    if not p.exists():
        return {"by_id": by_id, "by_name": by_name}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                uid = r["user_id"]
                by_id[uid] = r
                for key in ("username", "display_name"):
                    val = r.get(key)
                    if val:
                        by_name.setdefault(val.lower(), r)
            except (KeyError, json.JSONDecodeError):
                pass
    return {"by_id": by_id, "by_name": by_name}


def load_messages(guild_id: str) -> list[dict]:
    """Return every collected message for this guild, across all per-user files.

    Prefer load_user_messages() when working per class — that is the point of
    the per-user layout. This flat view exists for callers that genuinely need
    the whole guild (stats, migration, ad-hoc inspection).
    """
    out: list[dict] = []
    for uid in list_user_ids(guild_id):
        out.extend(load_user_messages(guild_id, uid))
    return out


def messages_by_user(guild_id: str) -> dict[str, list[dict]]:
    """Return {user_id: [messages]} — one entry per per-user file."""
    return {uid: load_user_messages(guild_id, uid) for uid in list_user_ids(guild_id)}


def user_message_counts(guild_id: str) -> dict[str, int]:
    """Return {user_id: message_count} without holding every message in memory."""
    counts = {}
    for uid in list_user_ids(guild_id):
        p = user_message_path(guild_id, uid)
        with p.open(encoding="utf-8") as f:
            counts[uid] = sum(1 for line in f if line.strip())
    return counts


def dataset_stats(guild_id: str = None) -> dict:
    """Return {guild_id: {messages, users, channels}} for one or all guilds."""
    if not _DATA_ROOT.exists():
        return {}
    roots = [_guild_dir(str(guild_id))] if guild_id else [d for d in _DATA_ROOT.iterdir() if d.is_dir()]
    stats = {}
    for d in roots:
        gid = d.name
        msg_count, user_ids, channel_ids = 0, set(), set()
        for uid in list_user_ids(gid):
            for r in load_user_messages(gid, uid):
                msg_count += 1
                user_ids.add(r.get("author_id", uid))
                channel_ids.add(r.get("channel_id"))
        stats[gid] = {"messages": msg_count, "users": len(user_ids), "channels": len(channel_ids)}
    return stats


def migrate_to_per_user(guild_id: str) -> dict:
    """Split a legacy messages.jsonl into users/{user_id}.jsonl.

    Idempotent: the legacy file is renamed to messages.jsonl.migrated on
    success, so a second run is a no-op. Returns a summary dict.
    """
    legacy = _guild_dir(guild_id) / "messages.jsonl"
    if not legacy.exists():
        return {"guild_id": str(guild_id), "migrated": 0, "users": 0, "skipped": "no messages.jsonl"}

    by_user: dict[str, list[dict]] = {}
    malformed = 0
    with legacy.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            by_user.setdefault(str(r.get("author_id") or "unknown"), []).append(r)

    # Merge rather than overwrite, so a partially-migrated guild stays correct.
    total = 0
    for uid, records in by_user.items():
        p = user_message_path(guild_id, uid)
        existing = {r.get("message_id") for r in load_user_messages(guild_id, uid)}
        with p.open("a", encoding="utf-8") as f:
            for r in records:
                if r.get("message_id") in existing:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1

    legacy.rename(legacy.with_suffix(".jsonl.migrated"))
    return {
        "guild_id":  str(guild_id),
        "migrated":  total,
        "users":     len(by_user),
        "malformed": malformed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Message storage utilities for user recognition collection."
    )
    parser.add_argument("--migrate", nargs="?", const="__ALL__", metavar="GUILD_ID",
                        help="Split legacy messages.jsonl into users/{user_id}.jsonl "
                             "(all guilds if no ID given)")
    parser.add_argument("--stats", action="store_true", help="Print per-guild stats")
    args = parser.parse_args()

    if args.migrate:
        gids = ([d.name for d in _DATA_ROOT.iterdir() if d.is_dir()]
                if args.migrate == "__ALL__" else [args.migrate])
        for gid in gids:
            result = migrate_to_per_user(gid)
            if result.get("skipped"):
                print(f"{gid}: {result['skipped']}")
            else:
                print(f"{gid}: {result['migrated']} message(s) -> "
                      f"{result['users']} per-user file(s)"
                      + (f", {result['malformed']} malformed line(s)" if result["malformed"] else ""))
    elif args.stats:
        for gid, s in dataset_stats().items():
            print(f"  {gid}  |  {s['messages']} messages  |  "
                  f"{s['users']} user(s)  |  {s['channels']} channel(s)")
    else:
        parser.print_help()
