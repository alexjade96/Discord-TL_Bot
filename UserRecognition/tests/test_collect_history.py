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

def _msg(mid="1", author="alice"):
    return {
        "message_id": mid,
        "guild_id": "guild1",
        "channel_id": "ch1",
        "channel_name": "general",
        "author_id": "100",
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
        lines = (tmp_path / "guild1" / "messages.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_deduplicates_by_message_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(collect_history, "_DATA_ROOT", tmp_path)
        seen = {"1"}  # already seen
        result = collect_history.save_message(_msg("1"), "guild1", seen)
        assert result is False
        # File should not exist
        assert not (tmp_path / "guild1" / "messages.jsonl").exists()

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
        lines = (tmp_path / "guild1" / "messages.jsonl").read_text().splitlines()
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
