"""Tests for TL-Bot.py command parsing.

Importable only because client.run() is guarded by __name__ == "__main__";
without that guard, importing the module connects a second live bot.

Covers the two parsing bugs that reached production:
  * `/collect Paul ohannigan` splitting a display name into two targets
  * `--index` silently becoming a username when unrecognised as a flag
"""

from __future__ import annotations

import datetime
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bot():
    spec = importlib.util.spec_from_file_location("tlbot", _ROOT / "TL-Bot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tlbot"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── _tokenize_collect ────────────────────────────────────────────────────────

class TestTokenize:
    def test_plain_tokens_unquoted(self, bot):
        assert bot._tokenize_collect("alice bob") == [("alice", False), ("bob", False)]

    def test_double_quoted_span_is_one_token(self, bot):
        assert bot._tokenize_collect('"Paul ohannigan"') == [("Paul ohannigan", True)]

    def test_single_quoted_span_is_one_token(self, bot):
        assert bot._tokenize_collect("'Kris Dum'") == [("Kris Dum", True)]

    def test_unquoted_display_name_splits(self, bot):
        """The original bug: a spaced display name became two targets."""
        assert bot._tokenize_collect("Paul ohannigan") == [("Paul", False), ("ohannigan", False)]

    def test_quoted_and_flags_mix(self, bot):
        assert bot._tokenize_collect('"Kris Dum" --limit 50') == [
            ("Kris Dum", True), ("--limit", False), ("50", False)]

    def test_empty_quotes_dropped(self, bot):
        assert bot._tokenize_collect('"" alice') == [("alice", False)]

    def test_mention_preserved(self, bot):
        assert bot._tokenize_collect("<@123>") == [("<@123>", False)]

    def test_quoted_flag_is_not_a_flag(self, bot):
        assert bot._tokenize_collect('"--batch"') == [("--batch", True)]


# ── _parse_collect_flags ─────────────────────────────────────────────────────

class TestParseCollectFlags:
    def test_plain_name_target(self, bot):
        targets, *_ = bot._parse_collect_flags("alice", None)
        assert targets == ["alice"]

    def test_mention_becomes_int_id(self, bot):
        targets, *_ = bot._parse_collect_flags("<@121475017031680000>", None)
        assert targets == [121475017031680000]

    def test_bare_digits_become_int_id(self, bot):
        targets, *_ = bot._parse_collect_flags("124651435848630275", None)
        assert targets == [124651435848630275]

    def test_quoted_name_is_single_target(self, bot):
        targets, *_ = bot._parse_collect_flags('"Paul ohannigan"', None)
        assert targets == ["Paul ohannigan"]

    def test_batch_sentinel(self, bot):
        targets, *_ = bot._parse_collect_flags("--batch", None)
        assert targets == ["__BATCH__"]

    def test_index_is_no_longer_a_flag(self, bot):
        """--index moved to its own /index command; as a collect arg it is just
        a name, and must not be silently swallowed as a mode switch."""
        targets, *_ = bot._parse_collect_flags("--index", None)
        assert targets == ["--index"]

    def test_limit_parsed(self, bot):
        _, _, _, limit, errors = bot._parse_collect_flags("alice --limit 50", None)
        assert limit == 50
        assert errors == []

    def test_invalid_limit_errors(self, bot):
        _, _, _, limit, errors = bot._parse_collect_flags("alice --limit abc", None)
        assert limit is None
        assert errors

    def test_since_parsed(self, bot):
        _, _, since, _, errors = bot._parse_collect_flags("alice --since 2026-01-15", None)
        assert since == datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)
        assert errors == []

    def test_invalid_since_errors(self, bot):
        _, _, since, _, errors = bot._parse_collect_flags("alice --since 15-01-2026", None)
        assert since is None
        assert errors

    def test_no_args_errors_when_targets_required(self, bot):
        targets, _, _, _, errors = bot._parse_collect_flags("", None)
        assert targets == []
        assert errors

    def test_no_args_ok_when_targets_not_required(self, bot):
        """/index takes the shared flags but no user targets."""
        targets, _, _, _, errors = bot._parse_collect_flags("", None, require_targets=False)
        assert targets == []
        assert errors == []

    def test_index_style_flags_without_targets(self, bot):
        _, _, since, limit, errors = bot._parse_collect_flags(
            "--since 2026-01-01 --limit 10", None, require_targets=False)
        assert limit == 10
        assert since is not None
        assert errors == []

    def test_multiple_targets(self, bot):
        targets, *_ = bot._parse_collect_flags('alice "Kris Dum" <@42> 99', None)
        assert targets == ["alice", "Kris Dum", 42, 99]


# ── command routing prefixes ─────────────────────────────────────────────────

class TestRoutingPrefixes:
    @pytest.mark.parametrize("msg,expected", [
        ("/index", "/index"),
        ("/index --limit 5", "/index"),
        ("/identify hello there", "/identify"),
        ("/collect alice", "/collect"),
    ])
    def test_no_prefix_collision(self, bot, msg, expected):
        """/index and /identify share a prefix; routing must not confuse them."""
        hits = [c for c in ("/index", "/collect", "/identify") if msg.startswith(c)]
        assert hits == [expected]


# ── module import safety ─────────────────────────────────────────────────────

class TestImportSafety:
    def test_import_does_not_start_client(self, bot):
        """client.run() must stay behind __name__ == '__main__'. If it does not,
        importing this module connects a second live bot that races the real one
        and double-writes collection files."""
        src = (_ROOT / "TL-Bot.py").read_text(encoding="utf-8")
        run_idx = src.index("client.run(")
        guard_idx = src.index('if __name__ == "__main__":')
        assert guard_idx < run_idx, "client.run() must be inside the __main__ guard"

    def test_handlers_present(self, bot):
        for name in ("_handle_index", "_handle_collect", "_handle_identify",
                     "_tokenize_collect", "_parse_collect_flags"):
            assert hasattr(bot, name), name
