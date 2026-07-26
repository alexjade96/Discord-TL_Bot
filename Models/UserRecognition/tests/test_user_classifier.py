"""Tests for user_classifier — tokenizer and torch model loading are mocked.

Covers the parts that run without downloading a backbone: dataset loading,
augmentation invariants, and deploy/remove filesystem behaviour.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from user_classifier import data as ac_data      # noqa: E402
from user_classifier import deploy as ac_deploy  # noqa: E402


class _FakeTokenizer:
    """Stands in for a HF tokenizer: returns fixed-length int tensors."""

    def __call__(self, text, truncation=True, max_length=8, padding='max_length',
                 return_tensors='pt'):
        import torch
        ids = [hash(text) % 100] * max_length
        return {
            'input_ids':      torch.tensor([ids]),
            'attention_mask': torch.tensor([[1] * max_length]),
        }


def _write_dataset(root, guild="g1", n=12):
    d = root / guild
    d.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", n), ("val", 4), ("test", 4)):
        with (d / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(count):
                f.write(json.dumps({
                    "text":  f"{split} sample {i} with some words in it",
                    "label": i % 2,
                    "author_id": f"u{i % 2}",
                    "username": ["alice", "bob"][i % 2],
                    "channel_id": "c1",
                    "timestamp": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                    "n_messages": 1,
                }) + "\n")
    (d / "label_map.json").write_text(json.dumps({
        "u0": {"label": 0, "username": "alice"},
        "u1": {"label": 1, "username": "bob"},
    }), encoding="utf-8")
    return d


# ── Dataset loading ──────────────────────────────────────────────────────────

class TestLoadSplits:
    def test_load_split_reads_rows(self, tmp_path):
        d = _write_dataset(tmp_path)
        rows = ac_data.load_split(d, "train")
        assert len(rows) == 12
        assert rows[0]["label"] in (0, 1)

    def test_load_split_missing_file_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            ac_data.load_split(d, "train")

    def test_label_map_orders_by_label_int(self, tmp_path):
        d = _write_dataset(tmp_path)
        assert ac_data.load_label_map(d) == ["alice", "bob"]

    def test_label_map_missing_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            ac_data.load_label_map(d)


class TestChatDataset:
    def test_returns_batch_dict_and_label(self, tmp_path):
        d = _write_dataset(tmp_path)
        rows = ac_data.load_split(d, "train")
        ds = ac_data.ChatDataset(rows, _FakeTokenizer(), max_length=8)
        item, label = ds[0]
        assert set(item) == {"input_ids", "attention_mask"}
        assert item["input_ids"].shape[0] == 8      # squeezed batch dim
        assert int(label) in (0, 1)

    def test_length_matches_rows(self, tmp_path):
        d = _write_dataset(tmp_path)
        rows = ac_data.load_split(d, "train")
        assert len(ac_data.ChatDataset(rows, _FakeTokenizer())) == len(rows)

    def test_dataloaders_cover_all_splits(self, tmp_path):
        d = _write_dataset(tmp_path)
        tr, va, te, names = ac_data.get_dataloaders(
            d, _FakeTokenizer(), batch_size=4, max_length=8, augment='none')
        assert names == ["alice", "bob"]
        assert len(tr.dataset) == 12
        assert len(va.dataset) == 4
        assert len(te.dataset) == 4

    def test_eval_splits_are_not_augmented(self, tmp_path):
        d = _write_dataset(tmp_path)
        _, va, te, _ = ac_data.get_dataloaders(
            d, _FakeTokenizer(), batch_size=4, max_length=8, augment='heavy')
        assert va.dataset.augment is None
        assert te.dataset.augment is None


# ── Augmentation ─────────────────────────────────────────────────────────────

class TestAugment:
    def test_none_is_identity(self):
        aug = ac_data.get_augment('none')
        assert aug("the quick brown fox") == "the quick brown fox"

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            ac_data.get_augment('extreme')

    def test_token_dropout_never_empties_text(self):
        drop = ac_data.TokenDropout(p=1.0)
        assert drop("a b c d e f").strip() != ""

    def test_token_dropout_leaves_short_text_alone(self):
        drop = ac_data.TokenDropout(p=1.0)
        assert drop("hi there") == "hi there"

    def test_span_dropout_leaves_short_text_alone(self):
        span = ac_data.SpanDropout()
        assert span("one two three") == "one two three"

    def test_span_dropout_shortens_long_text(self):
        span = ac_data.SpanDropout(max_frac=0.5)
        text = " ".join(f"w{i}" for i in range(40))
        assert len(span(text).split()) < 40


# ── Deploy ───────────────────────────────────────────────────────────────────

def _write_checkpoint_dir(root, guild="g1"):
    d = root / guild
    d.mkdir(parents=True, exist_ok=True)
    (d / "best.pt").write_bytes(b"fake-weights")
    (d / "config.json").write_text(json.dumps({
        "backbone": "xlm-roberta-base", "guild_id": guild,
        "num_classes": 2, "max_length": 256,
    }), encoding="utf-8")
    (d / "class_names.json").write_text(json.dumps(["alice", "bob"]), encoding="utf-8")
    return d


class TestDeploy:
    def test_installs_expected_files(self, tmp_path, monkeypatch):
        ckpts = tmp_path / "checkpoints"
        _write_checkpoint_dir(ckpts)
        model_root = tmp_path / "installed"
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", model_root)

        result = ac_deploy.deploy("g1", checkpoint_dir=ckpts / "g1")
        dest = model_root / "g1"
        for name in ("model.pt", "config.json", "class_names.json", "meta.json"):
            assert (dest / name).exists(), name
        assert result["meta"]["model_type"] == "neural"
        assert result["meta"]["backbone"] == "xlm-roberta-base"

    def test_missing_checkpoint_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", tmp_path / "installed")
        with pytest.raises(FileNotFoundError):
            ac_deploy.deploy("g1", checkpoint_dir=tmp_path / "nope")

    def test_reports_shadowed_tfidf_by_default(self, tmp_path, monkeypatch):
        ckpts = tmp_path / "checkpoints"
        _write_checkpoint_dir(ckpts)
        model_root = tmp_path / "installed"
        (model_root / "g1").mkdir(parents=True)
        for f in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
            (model_root / "g1" / f).write_bytes(b"")
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", model_root)

        result = ac_deploy.deploy("g1", checkpoint_dir=ckpts / "g1")
        assert len(result["shadowed_tfidf"]) == 3
        assert (model_root / "g1" / "clf.pkl").exists()

    def test_replace_tfidf_deletes_pickles(self, tmp_path, monkeypatch):
        ckpts = tmp_path / "checkpoints"
        _write_checkpoint_dir(ckpts)
        model_root = tmp_path / "installed"
        (model_root / "g1").mkdir(parents=True)
        for f in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
            (model_root / "g1" / f).write_bytes(b"")
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", model_root)

        result = ac_deploy.deploy("g1", checkpoint_dir=ckpts / "g1", keep_tfidf=False)
        assert result["shadowed_tfidf"] == []
        assert not (model_root / "g1" / "clf.pkl").exists()

    def test_remove_reverts_to_tfidf(self, tmp_path, monkeypatch):
        ckpts = tmp_path / "checkpoints"
        _write_checkpoint_dir(ckpts)
        model_root = tmp_path / "installed"
        (model_root / "g1").mkdir(parents=True)
        for f in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
            (model_root / "g1" / f).write_bytes(b"")
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", model_root)

        ac_deploy.deploy("g1", checkpoint_dir=ckpts / "g1")
        result = ac_deploy.remove("g1")
        assert result["fallback"] == "tfidf"
        assert not (model_root / "g1" / "model.pt").exists()
        assert (model_root / "g1" / "clf.pkl").exists()

    def test_remove_without_tfidf_has_no_fallback(self, tmp_path, monkeypatch):
        ckpts = tmp_path / "checkpoints"
        _write_checkpoint_dir(ckpts)
        model_root = tmp_path / "installed"
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", model_root)

        ac_deploy.deploy("g1", checkpoint_dir=ckpts / "g1")
        result = ac_deploy.remove("g1")
        assert result["fallback"] is None

    def test_remove_absent_guild_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac_deploy, "_MODEL_ROOT", tmp_path / "installed")
        assert ac_deploy.remove("nope") == {"removed": [], "fallback": None}


# ── Model builder (no weights downloaded) ────────────────────────────────────

class TestModelBuilder:
    def test_rejects_unknown_backbone(self):
        from user_classifier.model_builder import create_model, create_tokenizer
        with pytest.raises(ValueError):
            create_model("bert-base-uncased", num_classes=2)
        with pytest.raises(ValueError):
            create_tokenizer("bert-base-uncased")

    def test_masked_mean_respects_attention_mask(self):
        import torch
        from user_classifier.model_builder import _masked_mean
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask   = torch.tensor([[1, 1, 0]])          # third token is padding
        assert torch.allclose(_masked_mean(hidden, mask), torch.tensor([[2.0, 2.0]]))

    def test_both_backbones_are_offered(self):
        from user_classifier.model_builder import BACKBONE_CHOICES
        assert "xlm-roberta-base" in BACKBONE_CHOICES
        assert "distilbert-base-multilingual-cased" in BACKBONE_CHOICES
