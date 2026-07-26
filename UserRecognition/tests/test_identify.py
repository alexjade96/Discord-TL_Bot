"""Tests for identify.py — sklearn/scipy calls are mocked."""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import identify


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_payload(label_map=None, probs=None):
    """Return the (word_vec, char_vec, clf, label_map) payload of mocks."""
    if label_map is None:
        label_map = {0: "alice", 1: "bob"}
    if probs is None:
        probs = [0.3, 0.7]

    word_vec = MagicMock()
    char_vec  = MagicMock()
    clf       = MagicMock()
    clf.classes_ = list(label_map.keys())
    clf.predict_proba.return_value = [probs]
    return word_vec, char_vec, clf, label_map


def _mock_load(label_map=None, probs=None):
    """Return what _load() returns for a TF-IDF guild: (backend, payload)."""
    return "tfidf", _mock_payload(label_map, probs)


def _write_tfidf(root, guild="999"):
    d = root / guild
    d.mkdir(parents=True, exist_ok=True)
    for fname in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
        (d / fname).write_bytes(b"")
    return d


def _write_neural(root, guild="999"):
    d = root / guild
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"")
    (d / "config.json").write_text('{"backbone": "xlm-roberta-base"}')
    (d / "class_names.json").write_text('["alice", "bob"]')
    return d


# ── model_exists ─────────────────────────────────────────────────────────────

class TestModelExists:
    def test_returns_false_when_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path / "user recognition")
        assert identify.model_exists("999") is False

    def test_returns_false_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        (tmp_path / "999").mkdir()
        # No pkl files present
        assert identify.model_exists("999") is False

    def test_returns_true_when_all_files_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        d = tmp_path / "999"
        d.mkdir()
        for fname in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
            (d / fname).write_bytes(b"")
        assert identify.model_exists("999") is True

    def test_requires_all_three_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        d = tmp_path / "999"
        d.mkdir()
        (d / "word_vec.pkl").write_bytes(b"")
        (d / "char_vec.pkl").write_bytes(b"")
        # clf.pkl missing
        assert identify.model_exists("999") is False


# ── _load (cache behaviour) ───────────────────────────────────────────────────

class TestLoad:
    def test_raises_when_no_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path / "user recognition")
        monkeypatch.setitem(identify._cache, "missing", None)
        identify._cache.pop("missing", None)
        with pytest.raises(FileNotFoundError):
            identify._load("missing")

    def test_cache_hit_skips_pickle(self, monkeypatch):
        sentinel = ("tfidf", ("w", "c", "clf", {0: "alice"}))
        monkeypatch.setitem(identify._cache, "cached_guild", sentinel)
        result = identify._load("cached_guild")
        assert result is sentinel
        # Clean up
        identify._cache.pop("cached_guild", None)


# ── identify ──────────────────────────────────────────────────────────────────

class TestIdentify:
    @patch("identify._load")
    @patch("identify._hstack")
    def test_result_keys_present(self, mock_hstack, mock_load):
        mock_load.return_value = _mock_load()
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "username" in r
            assert "label" in r
            assert "score" in r

    @patch("identify._load")
    @patch("identify._hstack")
    def test_sorted_descending_by_score(self, mock_hstack, mock_load):
        mock_load.return_value = _mock_load(probs=[0.3, 0.7])
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("identify._load")
    @patch("identify._hstack")
    def test_top_result_matches_highest_prob(self, mock_hstack, mock_load):
        label_map = {0: "alice", 1: "bob"}
        mock_load.return_value = _mock_load(label_map=label_map, probs=[0.2, 0.8])
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        assert results[0]["username"] == "bob"
        assert results[0]["score"] == pytest.approx(0.8)

    @patch("identify._load")
    @patch("identify._hstack")
    def test_scores_sum_to_one(self, mock_hstack, mock_load):
        mock_load.return_value = _mock_load(probs=[0.3, 0.7])
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        total = sum(r["score"] for r in results)
        assert total == pytest.approx(1.0, abs=0.01)

    @patch("identify._load")
    @patch("identify._hstack")
    def test_username_resolved_from_label_map(self, mock_hstack, mock_load):
        label_map = {0: "charlie", 1: "dana"}
        mock_load.return_value = _mock_load(label_map=label_map, probs=[0.6, 0.4])
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        usernames = [r["username"] for r in results]
        assert "charlie" in usernames
        assert "dana" in usernames

    @patch("identify._load")
    @patch("identify._hstack")
    def test_label_fallback_when_not_in_map(self, mock_hstack, mock_load):
        """If a label int has no entry in label_map, falls back to str(label)."""
        word_vec = MagicMock()
        char_vec  = MagicMock()
        clf       = MagicMock()
        clf.classes_ = [0, 1]          # model knows about both labels
        clf.predict_proba.return_value = [[0.4, 0.6]]
        label_map = {0: "alice"}       # label 1 intentionally absent from map
        mock_load.return_value = ("tfidf", (word_vec, char_vec, clf, label_map))
        mock_hstack.return_value = MagicMock()
        results = identify.identify("hello", "guild1")
        usernames = [r["username"] for r in results]
        assert "1" in usernames        # fallback to str(1)

    @patch("identify._load")
    @patch("identify._hstack")
    def test_passes_text_to_vectorizers(self, mock_hstack, mock_load):
        word_vec, char_vec, clf, label_map = _mock_payload()
        mock_load.return_value = ("tfidf", (word_vec, char_vec, clf, label_map))
        mock_hstack.return_value = MagicMock()
        identify.identify("test input", "guild1")
        word_vec.transform.assert_called_once_with(["test input"])
        char_vec.transform.assert_called_once_with(["test input"])

    def test_raises_when_no_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path / "user recognition")
        monkeypatch.setitem(identify._cache, "no_model", None)
        identify._cache.pop("no_model", None)
        with pytest.raises(FileNotFoundError):
            identify.identify("hello", "no_model")


# ── backend dispatch (tfidf vs neural) ────────────────────────────────────────

class TestModelType:
    def test_none_when_nothing_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        assert identify.model_type("999") is None

    def test_tfidf_when_only_pickles_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        _write_tfidf(tmp_path)
        assert identify.model_type("999") == "tfidf"

    def test_neural_when_only_checkpoint_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        _write_neural(tmp_path)
        assert identify.model_type("999") == "neural"

    def test_neural_wins_when_both_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        _write_tfidf(tmp_path)
        _write_neural(tmp_path)
        assert identify.model_type("999") == "neural"

    def test_falls_back_to_tfidf_when_neural_incomplete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        d = _write_tfidf(tmp_path)
        (d / "model.pt").write_bytes(b"")   # config.json + class_names.json missing
        assert identify.model_type("999") == "tfidf"

    def test_model_exists_true_for_either_backend(self, tmp_path, monkeypatch):
        monkeypatch.setattr(identify, "_MODEL_ROOT", tmp_path)
        _write_neural(tmp_path, "neural_guild")
        _write_tfidf(tmp_path, "tfidf_guild")
        assert identify.model_exists("neural_guild") is True
        assert identify.model_exists("tfidf_guild") is True
        assert identify.model_exists("absent_guild") is False


class TestNeuralDispatch:
    """identify() must route to the neural path without touching scipy/sklearn."""

    @patch("identify._load")
    def test_neural_payload_routes_to_neural_branch(self, mock_load):
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        with patch("identify._identify_neural") as mock_neural:
            mock_neural.return_value = [
                {"username": "bob", "label": 1, "score": 0.9},
                {"username": "alice", "label": 0, "score": 0.1},
            ]
            mock_load.return_value = ("neural", (model, tokenizer, ["alice", "bob"], 256, "cpu"))
            results = identify.identify("hello", "guild1")

        mock_neural.assert_called_once()
        assert results[0]["username"] == "bob"

    @patch("identify._load")
    @patch("identify._hstack")
    def test_tfidf_payload_does_not_call_neural(self, mock_hstack, mock_load):
        mock_hstack.return_value = MagicMock()
        mock_load.return_value = _mock_load()
        with patch("identify._identify_neural") as mock_neural:
            identify.identify("hello", "guild1")
        mock_neural.assert_not_called()

    @patch("identify._load")
    def test_neural_results_sorted_descending(self, mock_load):
        model = MagicMock()
        tokenizer = MagicMock()
        with patch("identify._identify_neural") as mock_neural:
            mock_neural.return_value = [
                {"username": "alice", "label": 0, "score": 0.2},
                {"username": "bob",   "label": 1, "score": 0.5},
                {"username": "carol", "label": 2, "score": 0.3},
            ]
            mock_load.return_value = ("neural", (model, tokenizer, ["alice", "bob", "carol"], 256, "cpu"))
            results = identify.identify("hello", "guild1")

        assert [r["score"] for r in results] == [0.5, 0.3, 0.2]
