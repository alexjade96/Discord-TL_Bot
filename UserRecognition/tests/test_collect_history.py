"""Tests for collect_history.py — all I/O uses tmp_path, no network calls."""

from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "0-Data", "training"))
import collect_history


# ── normalize_content ─────────────────────────────────────────────────────────

class TestNormalizeContent:
    def test_user_mention(self):
        assert collect_history.normalize_content("<@123456>") == "[USER]"

    def test_user_mention_with_bang(self):
        assert collect_history.normalize_content("<@!123456>") == "[USER]"

    def test_channel_mention(self):
        assert collect_history.normalize_content("<#123456>") == "[CHANNEL]"

    def test_role_mention(self):
        assert collect_history.normalize_content("<@&123456>") == "[ROLE]"

    def test_url(self):
        assert collect_history.normalize_content("https://example.com/path") == "[URL]"

    def test_custom_emoji(self):
        assert collect_history.normalize_content("<:wave:123456>") == "[EMOJI]"

    def test_animated_emoji(self):
        assert collect_history.normalize_content("<a:wave:123456>") == "[EMOJI]"

    def test_mixed_content(self):
        result = collect_history.normalize_content("hello <@123> check <#456>")
        assert result == "hello [USER] check [CHANNEL]"

    def test_plain_text_unchanged(self):
        assert collect_history.normalize_content("hello world") == "hello world"

    def test_strips_whitespace(self):
        assert collect_history.normalize_content("  hello  ") == "hello"


# ── save_message / load_seen_ids / load_messages ──────────────────────────────

def _msg(mid="1", author="alice", author_id="100"):
    return {
        "message_id": mid,
        "guild_id": "guild1",
        "channel_id": "ch1",
        "channel_name": "general",
        "author_id": author_id,
        "author_name": author,
        "content_raw": "hello",
        "content_normalized": "hello",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "edited_timestamp": None,
        "is_reply": False,
        "reply_to_author_id": None,
        "reply_to_author_name": None,
        "has_attachments": False,
        "has_embeds": False,
        "token_count": 1,
    }


class TestSaveMessage:
    def test_saves_new_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        result = collect_history.save_message(_msg("1"), "guild1", seen)
        assert result is True
        assert "1" in seen
        lines = (tmp_path / "guild1" / "users" / "100.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_deduplicates_by_message_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = {"1"}  # already seen
        result = collect_history.save_message(_msg("1"), "guild1", seen)
        assert result is False
        # File should not exist
        assert not (tmp_path / "guild1" / "users" / "100.jsonl").exists()

    def test_returns_false_on_duplicate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1"), "guild1", seen)
        result = collect_history.save_message(_msg("1"), "guild1", seen)
        assert result is False

    def test_multiple_messages_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1"), "guild1", seen)
        collect_history.save_message(_msg("2"), "guild1", seen)
        lines = (tmp_path / "guild1" / "users" / "100.jsonl").read_text().splitlines()
        assert len(lines) == 2


class TestLoadSeenIds:
    def test_returns_empty_set_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        result = collect_history.load_seen_ids("guild1")
        assert result == set()

    def test_returns_existing_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1"), "guild1", seen)
        collect_history.save_message(_msg("2"), "guild1", seen)
        result = collect_history.load_seen_ids("guild1")
        assert result == {"1", "2"}


class TestLoadMessages:
    def test_returns_empty_list_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        result = collect_history.load_messages("guild1")
        assert result == []

    def test_returns_all_messages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1"), "guild1", seen)
        collect_history.save_message(_msg("2"), "guild1", seen)
        result = collect_history.load_messages("guild1")
        assert len(result) == 2
        ids = {m["message_id"] for m in result}
        assert ids == {"1", "2"}


# ── save_user ─────────────────────────────────────────────────────────────────

class TestSaveUser:
    def _user(self, uid="100", first="2026-01-01T00:00:00+00:00"):
        return {"user_id": uid, "guild_id": "guild1", "username": "alice",
                "display_name": "Alice", "message_count": 10,
                "first_message_at": first, "last_message_at": "2026-07-01T00:00:00+00:00",
                "last_collected_at": "2026-07-01T00:00:00+00:00"}

    def test_saves_new_user(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_user(self._user(), "guild1")
        lines = (tmp_path / "guild1" / "users.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_upserts_existing_user(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_user(self._user(), "guild1")
        updated = self._user()
        updated["message_count"] = 20
        collect_history.save_user(updated, "guild1")
        lines = (tmp_path / "guild1" / "users.jsonl").read_text().splitlines()
        assert len(lines) == 1  # still one user
        data = json.loads(lines[0])
        assert data["message_count"] == 20

    def test_preserves_earliest_first_message_at(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_user(self._user(first="2026-06-01T00:00:00+00:00"), "guild1")
        later = self._user(first="2026-07-01T00:00:00+00:00")
        collect_history.save_user(later, "guild1")
        data = json.loads((tmp_path / "guild1" / "users.jsonl").read_text())
        assert data["first_message_at"] == "2026-06-01T00:00:00+00:00"


# ── save_guild / save_identity / load_identity ────────────────────────────────

class TestSaveGuild:
    def test_creates_guilds_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_guild("123", "Test Server", "2026-01-01T00:00:00+00:00")
        p = tmp_path / "guilds.jsonl"
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["guild_id"] == "123"
        assert data["guild_name"] == "Test Server"

    def test_upserts_existing_guild(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_guild("123", "Old Name", "2026-01-01T00:00:00+00:00")
        collect_history.save_guild("123", "New Name", "2026-07-01T00:00:00+00:00")
        lines = (tmp_path / "guilds.jsonl").read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["guild_name"] == "New Name"


class TestIdentity:
    def _records(self):
        return [
            {"user_id": "1", "guild_id": "g", "username": "alice",
             "display_name": "Alice", "indexed_at": "2026-01-01T00:00:00+00:00"},
            {"user_id": "2", "guild_id": "g", "username": "bob",
             "display_name": "Bob The Builder", "indexed_at": "2026-01-01T00:00:00+00:00"},
        ]

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_identity(self._records(), "g")
        result = collect_history.load_identity("g")
        assert "1" in result["by_id"]
        assert "2" in result["by_id"]

    def test_lookup_by_username(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_identity(self._records(), "g")
        result = collect_history.load_identity("g")
        assert result["by_name"]["alice"]["user_id"] == "1"

    def test_lookup_by_display_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_identity(self._records(), "g")
        result = collect_history.load_identity("g")
        assert result["by_name"]["bob the builder"]["user_id"] == "2"

    def test_case_insensitive_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_identity(self._records(), "g")
        result = collect_history.load_identity("g")
        assert "alice" in result["by_name"]
        assert result["by_name"].get("ALICE") is None  # keys are stored lowercase

    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        result = collect_history.load_identity("no_guild")
        assert result == {"by_id": {}, "by_name": {}}

    def test_overwrites_on_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_identity(self._records(), "g")
        collect_history.save_identity([self._records()[0]], "g")  # only alice
        result = collect_history.load_identity("g")
        assert "1" in result["by_id"]
        assert "2" not in result["by_id"]


# ── Author index (authors.jsonl) ──────────────────────────────────────────────

class TestAuthors:
    """authors.jsonl accumulates every author ever seen posting, including
    users who have since left the server — unlike identity.jsonl, which is a
    snapshot of current members and is overwritten on every scan."""

    def _records(self):
        return [
            {"user_id": "1", "guild_id": "g", "username": "alice",
             "display_name": "Alice A", "is_bot": False,
             "first_seen_at": "2026-01-02T00:00:00", "last_seen_at": "2026-01-05T00:00:00"},
            {"user_id": "2", "guild_id": "g", "username": "departed",
             "display_name": "Gone Guy", "is_bot": False,
             "first_seen_at": "2026-01-01T00:00:00", "last_seen_at": "2026-01-03T00:00:00"},
        ]

    def test_saves_and_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        new = collect_history.save_authors(self._records(), "g")
        assert new == 2
        result = collect_history.load_authors("g")
        assert set(result["by_id"]) == {"1", "2"}

    def test_lookup_by_username_and_display_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_authors(self._records(), "g")
        result = collect_history.load_authors("g")
        assert result["by_name"]["departed"]["user_id"] == "2"
        assert result["by_name"]["gone guy"]["user_id"] == "2"

    def test_accumulates_across_saves(self, tmp_path, monkeypatch):
        """Second save must not drop authors from the first — this is the
        property that makes a departed user resolvable later."""
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_authors(self._records(), "g")
        new = collect_history.save_authors([
            {"user_id": "3", "guild_id": "g", "username": "carol",
             "display_name": "Carol", "is_bot": False,
             "first_seen_at": "2026-02-01T00:00:00", "last_seen_at": "2026-02-01T00:00:00"},
        ], "g")
        assert new == 1
        result = collect_history.load_authors("g")
        assert set(result["by_id"]) == {"1", "2", "3"}

    def test_reports_zero_new_on_repeat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_authors(self._records(), "g")
        assert collect_history.save_authors(self._records(), "g") == 0

    def test_keeps_earliest_first_seen(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_authors(self._records(), "g")
        collect_history.save_authors([
            {"user_id": "1", "guild_id": "g", "username": "alice",
             "display_name": "Alice A", "is_bot": False,
             "first_seen_at": "2026-03-01T00:00:00", "last_seen_at": "2026-03-01T00:00:00"},
        ], "g")
        result = collect_history.load_authors("g")
        assert result["by_id"]["1"]["first_seen_at"] == "2026-01-02T00:00:00"
        assert result["by_id"]["1"]["last_seen_at"] == "2026-03-01T00:00:00"

    def test_skips_records_without_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        assert collect_history.save_authors([{"username": "nobody"}], "g") == 0

    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        assert collect_history.load_authors("no_guild") == {"by_id": {}, "by_name": {}}


# ── Per-user message storage ──────────────────────────────────────────────────

class TestPerUserStorage:
    """Messages are stored one file per user (users/{user_id}.jsonl) so each
    file is already a clean per-class corpus for training."""

    def test_routes_messages_to_per_user_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1", author_id="100"), "g", seen)
        collect_history.save_message(_msg("2", author_id="100"), "g", seen)
        collect_history.save_message(_msg("3", author_id="200"), "g", seen)
        assert (tmp_path / "g" / "users" / "100.jsonl").exists()
        assert (tmp_path / "g" / "users" / "200.jsonl").exists()
        assert len(collect_history.load_user_messages("g", "100")) == 2
        assert len(collect_history.load_user_messages("g", "200")) == 1

    def test_no_single_messages_file_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        collect_history.save_message(_msg("1"), "g", set())
        assert not (tmp_path / "g" / "messages.jsonl").exists()

    def test_list_user_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1", author_id="100"), "g", seen)
        collect_history.save_message(_msg("2", author_id="200"), "g", seen)
        assert collect_history.list_user_ids("g") == ["100", "200"]

    def test_list_user_ids_empty_when_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        assert collect_history.list_user_ids("nope") == []

    def test_seen_ids_span_all_users(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1", author_id="100"), "g", seen)
        collect_history.save_message(_msg("2", author_id="200"), "g", seen)
        assert collect_history.load_seen_ids("g") == {"1", "2"}

    def test_messages_by_user_groups_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        collect_history.save_message(_msg("1", author_id="100"), "g", seen)
        collect_history.save_message(_msg("2", author_id="200"), "g", seen)
        grouped = collect_history.messages_by_user("g")
        assert set(grouped) == {"100", "200"}
        assert len(grouped["100"]) == 1

    def test_user_message_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = set()
        for i in range(3):
            collect_history.save_message(_msg(str(i), author_id="100"), "g", seen)
        collect_history.save_message(_msg("9", author_id="200"), "g", seen)
        assert collect_history.user_message_counts("g") == {"100": 3, "200": 1}

    def test_missing_author_id_goes_to_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        rec = _msg("1")
        rec["author_id"] = ""
        collect_history.save_message(rec, "g", set())
        assert (tmp_path / "g" / "users" / "unknown.jsonl").exists()


class TestMigration:
    def _legacy(self, tmp_path, guild="g"):
        d = tmp_path / guild
        d.mkdir(parents=True, exist_ok=True)
        with (d / "messages.jsonl").open("w", encoding="utf-8") as f:
            for i in range(4):
                f.write(json.dumps(_msg(str(i), author_id="100" if i < 3 else "200")) + "\n")
        return d

    def test_splits_legacy_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        self._legacy(tmp_path)
        result = collect_history.migrate_to_per_user("g")
        assert result["migrated"] == 4
        assert result["users"] == 2
        assert len(collect_history.load_user_messages("g", "100")) == 3
        assert len(collect_history.load_user_messages("g", "200")) == 1

    def test_renames_legacy_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        self._legacy(tmp_path)
        collect_history.migrate_to_per_user("g")
        assert not (tmp_path / "g" / "messages.jsonl").exists()
        assert (tmp_path / "g" / "messages.jsonl.migrated").exists()

    def test_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        self._legacy(tmp_path)
        collect_history.migrate_to_per_user("g")
        second = collect_history.migrate_to_per_user("g")
        assert second["migrated"] == 0
        assert len(collect_history.load_user_messages("g", "100")) == 3

    def test_no_legacy_file_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        result = collect_history.migrate_to_per_user("g")
        assert result["migrated"] == 0
        assert result["skipped"]

    def test_counts_malformed_lines(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        d = tmp_path / "g"
        d.mkdir(parents=True)
        with (d / "messages.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(_msg("1")) + "\n")
            f.write("{not json\n")
        result = collect_history.migrate_to_per_user("g")
        assert result["migrated"] == 1
        assert result["malformed"] == 1
