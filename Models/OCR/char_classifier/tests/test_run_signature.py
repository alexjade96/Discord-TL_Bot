"""Tests for train.py's checkpoint-archiving safety net -- pure filesystem
and JSON logic, no model/GPU/network involved."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from char_classifier import train


def make_args(**overrides):
    base = dict(
        backbone='dinov2_vits14', epochs=24, freeze_epochs=3, scheduler='cosine',
        mixup_alpha=0.2, unfreeze_blocks=4, batch_size=64, lr=3e-4, augment='heavy',
        grid_mode='single', clip_grad=1.0, select_metric='val_acc', max_per_class=0,
        min_per_class=5, seed=42, no_weighted_sampler=False, resume=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def fixed_git_hash(monkeypatch):
    monkeypatch.setattr(train, '_git_commit_hash', lambda: 'aaa1111111')


def write_prior_run(ckpt_dir, config: dict, completed_epoch=23, with_last_pt=True):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'config.json').write_text(json.dumps(config))
    (ckpt_dir / 'progress.json').write_text(json.dumps({'completed': completed_epoch}))
    if with_last_pt:
        (ckpt_dir / 'last.pt').write_bytes(b'fake')


class TestScriptTag:
    def test_single(self):
        assert train._script_tag(['latin']) == 'latin'

    def test_all(self):
        assert train._script_tag(['latin', 'kana', 'hangul', 'cjk']) == 'all'

    def test_subset(self):
        assert train._script_tag(['latin', 'kana']) == 'kana_latin'


class TestBuildSignature:
    def test_shape(self):
        sig = train._build_signature(make_args(), ['latin'])
        assert set(sig) == set(train._SIGNATURE_KEYS) | {'git_commit'}
        assert sig['scripts'] == ['latin']
        assert sig['weighted_sampler'] is True
        assert sig['git_commit'] == 'aaa1111111'

    def test_no_weighted_sampler_inverted(self):
        sig = train._build_signature(make_args(no_weighted_sampler=True), ['latin'])
        assert sig['weighted_sampler'] is False


class TestArchiveDecision:
    def test_no_prior_artifacts_no_archive(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        ckpt_dir.mkdir(parents=True)
        args = make_args(resume=None)
        sig = train._build_signature(args, ['latin'])
        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)
        assert not (ckpt_dir.parent / 'archive').exists()

    def test_resume_matching_signature_no_archive(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        args = make_args(resume=str(ckpt_dir / 'last.pt'))
        sig = train._build_signature(args, ['latin'])
        write_prior_run(ckpt_dir, sig)

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        assert not (ckpt_dir.parent / 'archive').exists()
        assert (ckpt_dir / 'last.pt').exists()
        assert args.resume == str(ckpt_dir / 'last.pt')  # untouched

    def test_resume_mismatched_signature_archives_and_clears_resume(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        old_config = {**train._build_signature(make_args(), ['latin']),
                      'grid_mode': 'all', 'git_commit': 'aaa1111111'}
        write_prior_run(ckpt_dir, old_config, completed_epoch=23)

        args = make_args(resume=str(ckpt_dir / 'last.pt'), grid_mode='single')
        new_sig = train._build_signature(args, ['latin'])  # grid_mode='single' now

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, new_sig)

        archive_root = ckpt_dir.parent / 'archive' / 'latin'
        runs = list(archive_root.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]
        assert run_dir.name.endswith('_run1_epoch23')

        # old files moved out of ckpt_dir
        assert not (ckpt_dir / 'last.pt').exists()
        assert not (ckpt_dir / 'config.json').exists()
        assert (run_dir / 'last.pt').exists()
        assert (run_dir / 'config.json').exists()

        reason = json.loads((run_dir / 'archive.json').read_text())
        assert reason['reason'] == 'resume_signature_mismatch'
        fields = {m['field'] for m in reason['mismatched_fields']}
        assert 'grid_mode' in fields

        # nothing left to resume from
        assert args.resume is None

    def test_fresh_restart_always_archives_even_if_matching(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        args = make_args(resume=None)
        sig = train._build_signature(args, ['latin'])
        write_prior_run(ckpt_dir, sig, completed_epoch=23)  # identical config

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        archive_root = ckpt_dir.parent / 'archive' / 'latin'
        runs = list(archive_root.iterdir())
        assert len(runs) == 1
        reason = json.loads((runs[0] / 'archive.json').read_text())
        assert reason['reason'] == 'fresh_restart'
        assert not (ckpt_dir / 'last.pt').exists()

    def test_legacy_config_resume_no_archive(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        legacy_config = {'backbone': 'dinov2_vits14', 'scripts': ['latin'],
                          'epochs': 24, 'freeze_epochs': 3, 'scheduler': 'cosine',
                          'mixup_alpha': 0.2}  # no 'git_commit' -> predates this feature
        write_prior_run(ckpt_dir, legacy_config, completed_epoch=23)

        args = make_args(resume=str(ckpt_dir / 'last.pt'))
        sig = train._build_signature(args, ['latin'])

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        assert not (ckpt_dir.parent / 'archive').exists()
        assert (ckpt_dir / 'last.pt').exists()
        assert args.resume is not None

    def test_legacy_config_fresh_restart_archives(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        legacy_config = {'backbone': 'dinov2_vits14', 'scripts': ['latin'],
                          'epochs': 24, 'freeze_epochs': 3, 'scheduler': 'cosine',
                          'mixup_alpha': 0.2}
        write_prior_run(ckpt_dir, legacy_config, completed_epoch=23)

        args = make_args(resume=None)
        sig = train._build_signature(args, ['latin'])

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        archive_root = ckpt_dir.parent / 'archive' / 'latin'
        assert len(list(archive_root.iterdir())) == 1
        assert not (ckpt_dir / 'last.pt').exists()

    def test_run_number_increments(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        args = make_args(resume=None)
        sig = train._build_signature(args, ['latin'])

        write_prior_run(ckpt_dir, sig, completed_epoch=5)
        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        write_prior_run(ckpt_dir, sig, completed_epoch=9)
        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        archive_root = ckpt_dir.parent / 'archive' / 'latin'
        names = sorted(p.name for p in archive_root.iterdir())
        assert len(names) == 2
        assert '_run1_epoch5' in names[0]
        assert '_run2_epoch9' in names[1]

    def test_existing_manual_archive_dirs_untouched(self, tmp_path):
        ckpt_dir = tmp_path / 'checkpoints' / 'latin'
        args = make_args(resume=None)
        sig = train._build_signature(args, ['latin'])
        write_prior_run(ckpt_dir, sig, completed_epoch=23)

        manual_backup = ckpt_dir / 'backup_run1_epoch34_20260807'
        manual_backup.mkdir()
        (manual_backup / 'best.pt').write_bytes(b'fake')
        old_dir = ckpt_dir / 'old'
        old_dir.mkdir()

        train._check_and_archive_stale_run(ckpt_dir, ['latin'], args, sig)

        assert manual_backup.exists()
        assert (manual_backup / 'best.pt').exists()
        assert old_dir.exists()
