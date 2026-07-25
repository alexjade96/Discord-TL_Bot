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


def load_seen_ids(guild_id: str) -> set:
    """Return set of message_ids already stored for this guild."""
    p = _guild_dir(guild_id) / "messages.jsonl"
    if not p.exists():
        return set()
    seen = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["message_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return seen


def save_message(record: dict, guild_id: str, seen_ids: set) -> bool:
    """Append record to messages.jsonl if not already present. Returns True if new."""
    mid = record["message_id"]
    if mid in seen_ids:
        return False
    p = _guild_dir(guild_id) / "messages.jsonl"
    with p.open("a", encoding="utf-8") as f:
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


def load_messages(guild_id: str) -> list[dict]:
    """Return all collected messages for this guild as a list of dicts."""
    p = _guild_dir(guild_id) / "messages.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_stats(guild_id: str = None) -> dict:
    """Return {guild_id: {messages, users, channels}} for one or all guilds."""
    if not _DATA_ROOT.exists():
        return {}
    roots = [_guild_dir(str(guild_id))] if guild_id else [d for d in _DATA_ROOT.iterdir() if d.is_dir()]
    stats = {}
    for d in roots:
        gid = d.name
        msg_count, user_ids, channel_ids = 0, set(), set()
        mp = d / "messages.jsonl"
        if mp.exists():
            with mp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        msg_count += 1
                        user_ids.add(r.get("author_id"))
                        channel_ids.add(r.get("channel_id"))
                    except json.JSONDecodeError:
                        pass
        stats[gid] = {"messages": msg_count, "users": len(user_ids), "channels": len(channel_ids)}
    return stats
