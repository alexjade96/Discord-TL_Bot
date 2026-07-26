"""Tests for Models/Datasets/build_chat.py — filesystem only, no network."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Datasets"))
import build_chat  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _msg(mid, author_id, author_name, text, ts, channel="chan1", tokens=None):
    return {
        "message_id":         str(mid),
        "guild_id":           "g1",
        "channel_id":         channel,
        "author_id":          author_id,
        "author_name":        author_name,
        "content_normalized": text,
        "timestamp":          ts,
        "token_count":        tokens if tokens is not None else len(text.split()),
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def source(tmp_path):
    """A source root with two humans and one bot, 30 messages each."""
    root = tmp_path / "src"
    gdir = root / "g1"

    msgs = []
    mid = 0
    for uid, name in (("u1", "alice"), ("u2", "bob"), ("u3", "helperbot")):
        for i in range(30):
            mid += 1
            msgs.append(_msg(
                mid, uid, name,
                f"{name} message number {i} with several words",
                f"2026-01-{i + 1:02d}T00:00:00+00:00",
            ))
    _write_jsonl(gdir / "messages.jsonl", msgs)
    _write_jsonl(gdir / "users.jsonl", [
        {"user_id": "u1", "username": "alice"},
        {"user_id": "u2", "username": "bob"},
        {"user_id": "u3", "username": "helperbot"},
    ])
    _write_jsonl(gdir / "identity.jsonl", [
        {"user_id": "u1", "username": "alice",     "bot": False},
        {"user_id": "u2", "username": "bob",       "bot": False},
        {"user_id": "u3", "username": "helperbot", "bot": True},
    ])
    return root


def _read(out_root, split):
    p = out_root / "g1" / f"{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Bot exclusion ────────────────────────────────────────────────────────────

class TestBotExclusion:
    def test_bots_dropped_by_default(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out)
        assert stats["authors"] == 2
        assert stats["dropped"]["bot"] == 30
        names = {r["username"] for r in _read(out, "train")}
        assert "helperbot" not in names

    def test_include_bots_keeps_them(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out, exclude_bots=False)
        assert stats["authors"] == 3

    def test_exclude_users_by_name(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out,
                                 exclude_bots=False, exclude_users=["helperbot"])
        assert stats["authors"] == 2

    def test_errors_when_too_few_authors(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out,
                                 exclude_users=["alice"])
        assert "error" in stats
        assert not (out / "g1").exists()


# ── Split integrity — the reason this builder exists ─────────────────────────

class TestSplitIntegrity:
    def test_no_text_shared_across_splits(self, source, tmp_path):
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        tr = {r["text"] for r in _read(out, "train")}
        va = {r["text"] for r in _read(out, "val")}
        te = {r["text"] for r in _read(out, "test")}
        assert tr & va == set()
        assert tr & te == set()
        assert va & te == set()

    def test_split_is_chronological_per_author(self, source, tmp_path):
        """Every train sample must predate every val sample, which must predate test."""
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        rows = {s: _read(out, s) for s in ("train", "val", "test")}
        for uid in ("u1", "u2"):
            tr = [r["timestamp"] for r in rows["train"] if r["author_id"] == uid]
            va = [r["timestamp"] for r in rows["val"]   if r["author_id"] == uid]
            te = [r["timestamp"] for r in rows["test"]  if r["author_id"] == uid]
            assert max(tr) <= min(va)
            assert max(va) <= min(te)

    def test_all_three_splits_populated(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out)
        assert stats["train"] > 0
        assert stats["val"] > 0
        assert stats["test"] > 0

    def test_every_author_present_in_train(self, source, tmp_path):
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        assert {r["author_id"] for r in _read(out, "train")} == {"u1", "u2"}

    def test_meta_records_chronological_strategy(self, source, tmp_path):
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        meta = json.loads((out / "g1" / "meta.json").read_text(encoding="utf-8"))
        assert meta["config"]["split_strategy"] == "chronological"


# ── Chunking ─────────────────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_one_is_one_sample_per_message(self):
        msgs = [_msg(i, "u1", "alice", f"text {i}", f"2026-01-{i:02d}T00:00:00+00:00")
                for i in range(1, 6)]
        assert len(build_chat.chunk_messages(msgs, 1)) == 5

    def test_chunk_groups_messages(self):
        msgs = [_msg(i, "u1", "alice", f"text{i}", f"2026-01-{i:02d}T00:00:00+00:00")
                for i in range(1, 9)]
        out = build_chat.chunk_messages(msgs, 4)
        assert len(out) == 2
        assert out[0]["n_messages"] == 4
        assert "text1" in out[0]["text"] and "text4" in out[0]["text"]

    def test_chunk_drops_tiny_trailing_group(self):
        msgs = [_msg(i, "u1", "alice", f"text{i}", f"2026-01-{i:02d}T00:00:00+00:00")
                for i in range(1, 10)]        # 9 messages, chunk 4 -> 4 + 4 + 1
        out = build_chat.chunk_messages(msgs, 4)
        assert len(out) == 2                  # the 1-message tail is dropped

    def test_chunk_keeps_substantial_trailing_group(self):
        msgs = [_msg(i, "u1", "alice", f"text{i}", f"2026-01-{i:02d}T00:00:00+00:00")
                for i in range(1, 12)]        # 11 messages, chunk 4 -> 4 + 4 + 3
        out = build_chat.chunk_messages(msgs, 4)
        assert len(out) == 3
        assert out[-1]["n_messages"] == 3

    def test_chunk_sorts_chronologically(self):
        msgs = [
            _msg(2, "u1", "alice", "second", "2026-01-02T00:00:00+00:00"),
            _msg(1, "u1", "alice", "first",  "2026-01-01T00:00:00+00:00"),
        ]
        out = build_chat.chunk_messages(msgs, 2)
        assert out[0]["text"] == "first second"

    def test_chunking_reduces_sample_count(self, source, tmp_path):
        out = tmp_path / "out"
        flat    = build_chat.build("g1", source_root=source, out_root=out, chunk=1)
        chunked = build_chat.build("g1", source_root=source, out_root=out, chunk=5)
        assert chunked["train"] < flat["train"]


# ── Filtering ────────────────────────────────────────────────────────────────

class TestFiltering:
    def test_min_tokens_applies_to_chunks_not_messages(self, tmp_path):
        """Short messages must survive into chunks rather than being dropped
        individually — filtering first discards ~80% of a real corpus."""
        root = tmp_path / "src"
        msgs = []
        for uid, name in (("u1", "alice"), ("u2", "bob")):
            for i in range(40):
                msgs.append(_msg(f"{uid}{i}", uid, name, "hi",   # 1 token each
                                 f"2026-01-{i + 1:02d}T00:00:00+00:00"))
        _write_jsonl(root / "g1" / "messages.jsonl", msgs)
        _write_jsonl(root / "g1" / "users.jsonl", [
            {"user_id": "u1", "username": "alice"}, {"user_id": "u2", "username": "bob"}])

        out = tmp_path / "out"
        # chunk 4 assembles 4 x "hi" = 4 tokens, clearing min_tokens 3.
        stats = build_chat.build("g1", source_root=root, out_root=out,
                                 chunk=4, min_tokens=3, min_messages=10)
        assert "error" not in stats
        assert stats["dropped"].get("short_chunks", 0) == 0
        assert stats["train"] > 0

    def test_chunks_below_min_tokens_are_dropped(self, tmp_path):
        root = tmp_path / "src"
        msgs = []
        for uid, name in (("u1", "alice"), ("u2", "bob")):
            for i in range(40):
                msgs.append(_msg(f"{uid}{i}", uid, name, "hi",
                                 f"2026-01-{i + 1:02d}T00:00:00+00:00"))
        _write_jsonl(root / "g1" / "messages.jsonl", msgs)
        _write_jsonl(root / "g1" / "users.jsonl", [
            {"user_id": "u1", "username": "alice"}, {"user_id": "u2", "username": "bob"}])

        out = tmp_path / "out"
        # chunk 2 gives 2-token samples, below min_tokens 5 -> nothing survives.
        stats = build_chat.build("g1", source_root=root, out_root=out,
                                 chunk=2, min_tokens=5, min_messages=10)
        assert "error" in stats
        assert "min-tokens" in stats["error"]

    def test_empty_messages_dropped(self, tmp_path):
        root = tmp_path / "src"
        msgs = []
        for uid, name in (("u1", "alice"), ("u2", "bob")):
            for i in range(25):
                text = "" if i < 5 else f"a longer message from {name} number {i}"
                msgs.append(_msg(f"{uid}{i}", uid, name, text,
                                 f"2026-01-{i + 1:02d}T00:00:00+00:00"))
        _write_jsonl(root / "g1" / "messages.jsonl", msgs)
        _write_jsonl(root / "g1" / "users.jsonl", [
            {"user_id": "u1", "username": "alice"}, {"user_id": "u2", "username": "bob"}])

        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=root, out_root=out,
                                 min_tokens=3, min_messages=10)
        assert stats["dropped"]["empty"] == 10

    def test_min_messages_drops_sparse_authors(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out, min_messages=100)
        assert "error" in stats

    def test_missing_guild_returns_error(self, tmp_path):
        stats = build_chat.build("nope", source_root=tmp_path / "src", out_root=tmp_path / "out")
        assert "error" in stats


# ── Outputs ──────────────────────────────────────────────────────────────────

class TestPerUserLayout:
    """Collection stores users/{user_id}.jsonl; a legacy messages.jsonl is still
    read so an unmigrated guild keeps working."""

    def _write_per_user(self, root, guild="g1"):
        d = root / guild / "users"
        for uid, name in (("u1", "alice"), ("u2", "bob")):
            rows = [
                _msg(f"{uid}{i}", uid, name, f"{name} message {i} with several words",
                     f"2026-01-{i + 1:02d}T00:00:00+00:00")
                for i in range(30)
            ]
            _write_jsonl(d / f"{uid}.jsonl", rows)
        _write_jsonl(root / guild / "users.jsonl", [
            {"user_id": "u1", "username": "alice"},
            {"user_id": "u2", "username": "bob"},
        ])

    def test_reads_per_user_files(self, tmp_path):
        root = tmp_path / "src"
        self._write_per_user(root)
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=root, out_root=out)
        assert stats["authors"] == 2
        assert stats["messages_in"] == 60

    def test_user_id_comes_from_filename(self, tmp_path):
        root = tmp_path / "src"
        self._write_per_user(root)
        out = tmp_path / "out"
        build_chat.build("g1", source_root=root, out_root=out)
        assert {r["author_id"] for r in _read(out, "train")} == {"u1", "u2"}

    def test_per_user_layout_wins_over_legacy(self, tmp_path):
        """When both exist, the per-user files are authoritative."""
        root = tmp_path / "src"
        self._write_per_user(root)
        _write_jsonl(root / "g1" / "messages.jsonl", [
            _msg("legacy1", "u9", "stale", "stale message that should be ignored",
                 "2020-01-01T00:00:00+00:00")
        ])
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=root, out_root=out)
        assert stats["messages_in"] == 60
        assert "u9" not in {r["author_id"] for r in _read(out, "train")}

    def test_falls_back_to_legacy_messages_file(self, source, tmp_path):
        """The `source` fixture writes only messages.jsonl — still builds."""
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out)
        assert "error" not in stats
        assert stats["authors"] == 2


class TestOutputs:
    def test_writes_all_expected_files(self, source, tmp_path):
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "label_map.json", "meta.json"):
            assert (out / "g1" / name).exists(), name

    def test_labels_are_contiguous_from_zero(self, source, tmp_path):
        out = tmp_path / "out"
        build_chat.build("g1", source_root=source, out_root=out)
        lm = json.loads((out / "g1" / "label_map.json").read_text(encoding="utf-8"))
        labels = sorted(v["label"] for v in lm.values())
        assert labels == list(range(len(labels)))

    def test_warns_about_few_authors(self, source, tmp_path):
        out = tmp_path / "out"
        stats = build_chat.build("g1", source_root=source, out_root=out)
        assert any("authors" in w for w in stats["warnings"])

    def test_holdout_channel_goes_entirely_to_test(self, tmp_path):
        root = tmp_path / "src"
        msgs = []
        for uid, name in (("u1", "alice"), ("u2", "bob")):
            for i in range(20):
                msgs.append(_msg(f"{uid}a{i}", uid, name, f"{name} says something {i}",
                                 f"2026-01-{i + 1:02d}T00:00:00+00:00", channel="keep"))
            for i in range(10):
                msgs.append(_msg(f"{uid}b{i}", uid, name, f"{name} elsewhere {i}",
                                 f"2026-02-{i + 1:02d}T00:00:00+00:00", channel="held"))
        _write_jsonl(root / "g1" / "messages.jsonl", msgs)
        _write_jsonl(root / "g1" / "users.jsonl", [
            {"user_id": "u1", "username": "alice"}, {"user_id": "u2", "username": "bob"}])

        out = tmp_path / "out"
        build_chat.build("g1", source_root=root, out_root=out, holdout_channel="held")
        assert {r["channel_id"] for r in _read(out, "test")} == {"held"}
        assert "held" not in {r["channel_id"] for r in _read(out, "train")}
        assert "held" not in {r["channel_id"] for r in _read(out, "val")}
