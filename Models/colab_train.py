#!/usr/bin/env python3
"""
colab_train.py -- Colab session setup + training launcher for char_classifier.

Usage (VS Code tunnel terminal or Colab shell cell):

    # First session — full setup then train
    python Models/colab_train.py

    # Subsequent sessions — skip clone/dataset sync, resume from last.pt
    python Models/colab_train.py --resume

    # Skip individual setup steps if already done this session
    python Models/colab_train.py --resume --skip-clone --skip-dataset

    # Smoke test (10 images/class, 4 epochs) before committing to a full run
    python Models/colab_train.py --smoke-test

Bootstrap (paste into a fresh Colab terminal before the repo is cloned):

    git clone https://github.com/alexjade96/Discord-TL_Bot /content/Discord-TL_Bot
    python /content/Discord-TL_Bot/Models/colab_train.py
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ============================================================
# CONFIG — edit these to match your setup
# ============================================================

# Google Drive folder where checkpoints and dataset zip are stored.
# Create this folder on Drive before the first run.
DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/TL-Bot"

# GitHub username — set this once, used to build REPO_URL below.
GITHUB_USERNAME = "alexjade96"

# GitHub repo URL — built from GITHUB_USERNAME; override the whole string if needed.
REPO_URL = f"https://github.com/{GITHUB_USERNAME}/Discord-TL_Bot.git"

# Where the repo is cloned on the Colab VM (fast local SSD).
REPO_DIR = "/content/Discord-TL_Bot"

# Path to a zipped copy of char-dataset on Drive.
# Upload char-dataset.zip to DRIVE_ROOT/ before the first session.
# Alternatively set to None to copy from DRIVE_ROOT/char-dataset/ directly.
DATASET_ZIP = f"{DRIVE_ROOT}/char-dataset.zip"

# Where checkpoints are written. Must be on Drive so they survive session end.
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

SCRIPTS       = ["latin"]        # latin | kana | hangul | cjk | all
EPOCHS        = 30
FREEZE_EPOCHS = 5                # head-only warm-up epochs before backbone fine-tune
UNFREEZE_BLOCKS = 4
BATCH_SIZE    = 64
BACKBONE      = "dinov2_vits14"  # dinov2_vits14 | dinov2_vitb14 | convnext_tiny
GRID_MODE     = "all"            # single | rotated | all
MIXUP_ALPHA   = 0.4
CLIP_GRAD     = 1.0


# ============================================================
# HELPERS
# ============================================================

def _run(cmd: str, cwd: str = None, check: bool = True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# SETUP STEPS
# ============================================================

def mount_drive():
    if os.path.ismount("/content/drive"):
        print("[setup] Drive already mounted.")
        return
    if not _in_colab():
        print("[setup] Not in Colab — skipping Drive mount.")
        return
    from google.colab import drive
    drive.mount("/content/drive")


def clone_or_update_repo():
    if (Path(REPO_DIR) / ".git").exists():
        print(f"[setup] Repo exists at {REPO_DIR} — pulling latest.")
        _run(f"git -C {REPO_DIR} pull --ff-only")
    else:
        print(f"[setup] Cloning {REPO_URL} -> {REPO_DIR}")
        if GITHUB_USERNAME == "YOUR_USERNAME":
            print("[setup] ERROR: Set GITHUB_USERNAME in colab_train.py before running.")
            sys.exit(1)
        _run(f"git clone {REPO_URL} {REPO_DIR}")


def install_deps():
    print("[setup] Checking / installing packages ...")
    # torch, torchvision, numpy, PIL, sklearn, tqdm are pre-installed on Colab.
    _run("pip install -q wordninja lingua-language-detector")


def sync_dataset():
    """
    Copy char-dataset from Drive to fast VM-local SSD.
    Skipped if all required script subdirs already exist locally.
    Prefers DATASET_ZIP; falls back to a plain directory at DRIVE_ROOT/char-dataset/.
    """
    local_root = Path(REPO_DIR) / "Models" / "Datasets" / "char-dataset"
    scripts_needed = (
        {"latin", "kana", "hangul", "cjk"} if "all" in SCRIPTS else set(SCRIPTS)
    )
    if all((local_root / s).is_dir() for s in scripts_needed):
        print(f"[setup] Dataset already present at {local_root} — skipping sync.")
        return

    zip_path = Path(DATASET_ZIP) if DATASET_ZIP else None
    if zip_path and zip_path.exists():
        print(f"[setup] Unzipping {zip_path} -> {local_root.parent} ...")
        local_root.parent.mkdir(parents=True, exist_ok=True)
        _run(f"unzip -q {shlex.quote(str(zip_path))} -d {shlex.quote(str(local_root.parent))}")
    else:
        drive_dir = Path(DRIVE_ROOT) / "char-dataset"
        if drive_dir.exists():
            print(f"[setup] Copying {drive_dir} -> {local_root} ...")
            if local_root.exists():
                shutil.rmtree(local_root)
            shutil.copytree(str(drive_dir), str(local_root))
        else:
            print(
                f"\n[setup] ERROR: No dataset found.\n"
                f"  Expected zip : {zip_path}\n"
                f"  Expected dir : {drive_dir}\n\n"
                f"  To fix: zip the char-dataset/ folder and upload it to Drive:\n"
                f"    Compress-Archive -Path Models\\Datasets\\char-dataset "
                f"-DestinationPath char-dataset.zip\n"
                f"  Then upload char-dataset.zip to {DRIVE_ROOT}/ on Google Drive."
            )
            sys.exit(1)


# ============================================================
# DATASET ZIP (run locally before first Colab session)
# ============================================================

def zip_dataset(output_path: str = None, scripts: list = None):
    """
    Zip Models/Datasets/char-dataset/ for upload to Drive.

    The archive always contains a top-level 'char-dataset/' folder so that
    sync_dataset()'s 'unzip -d <Datasets/>'' unpacks to the correct location.

    Run locally (Windows):
        python Models/colab_train.py --zip-dataset
        python Models/colab_train.py --zip-dataset --scripts latin kana
        python Models/colab_train.py --zip-dataset --zip-output D:/upload/char-dataset.zip

    Then upload the resulting zip to:
        My Drive/Colab Notebooks/tl-bot/char-dataset.zip
    """
    dataset_root = Path(__file__).parent / "Datasets" / "char-dataset"
    if not dataset_root.exists():
        print(f"[zip] ERROR: Dataset not found at {dataset_root}")
        sys.exit(1)

    # Filter to requested scripts only, or include all present subdirs
    if scripts and "all" not in scripts:
        subdirs = [dataset_root / s for s in scripts if (dataset_root / s).is_dir()]
        missing = [s for s in scripts if not (dataset_root / s).is_dir()]
        if missing:
            print(f"[zip] WARNING: script dirs not found and will be skipped: {missing}")
    else:
        subdirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    if not subdirs:
        print("[zip] ERROR: No script subdirectories found to zip.")
        sys.exit(1)

    if output_path is None:
        out = Path(__file__).parent.parent / "char-dataset.zip"
    else:
        out = Path(output_path)

    print(f"[zip] Source : {dataset_root}")
    print(f"[zip] Scripts: {[p.name for p in subdirs]}")
    print(f"[zip] Output : {out}")
    print("[zip] Zipping ...")

    total_files = sum(1 for d in subdirs for f in d.rglob("*") if f.is_file())
    written = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for subdir in subdirs:
            for file in sorted(subdir.rglob("*")):
                if file.is_file():
                    arcname = Path("char-dataset") / subdir.name / file.relative_to(subdir)
                    zf.write(file, arcname)
                    written += 1
                    if written % 5000 == 0:
                        print(f"  {written}/{total_files} files ...")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[zip] Done: {written} files, {size_mb:.1f} MB -> {out}")
    print(f"\n  Upload to Drive: My Drive/Colab Notebooks/tl-bot/char-dataset.zip")


# ============================================================
# TRAINING LAUNCHER
# ============================================================

def _last_pt_path() -> Path:
    """
    Auto-scope matches char_classifier.train's own scoping:
    single script -> checkpoints/<script>/last.pt
    multiple      -> checkpoints/last.pt
    """
    base = Path(CKPT_DIR)
    if len(SCRIPTS) == 1 and SCRIPTS[0] != "all":
        return base / SCRIPTS[0] / "last.pt"
    return base / "last.pt"


def train(resume: bool, smoke_test: bool):
    ocr_dir = Path(REPO_DIR) / "Models" / "OCR"
    last_pt = _last_pt_path()

    scripts_arg = SCRIPTS if "all" not in SCRIPTS else ["all"]

    cmd = [
        sys.executable, "-u", "-m", "char_classifier.train",
        "--scripts",        *scripts_arg,
        "--epochs",         str(EPOCHS),
        "--freeze-epochs",  str(FREEZE_EPOCHS),
        "--unfreeze-blocks", str(UNFREEZE_BLOCKS),
        "--batch-size",     str(BATCH_SIZE),
        "--backbone",       BACKBONE,
        "--grid-mode",      GRID_MODE,
        "--mixup-alpha",    str(MIXUP_ALPHA),
        "--clip-grad",      str(CLIP_GRAD),
        "--checkpoint-dir", CKPT_DIR,
        "--no-tensorboard",
    ]

    if smoke_test:
        print("[train] Smoke-test mode: --max-per-class 10, --epochs 4, --freeze-epochs 2")
        cmd += ["--max-per-class", "10"]
        # patch epochs inline without modifying globals
        for flag in ("--epochs", "--freeze-epochs"):
            idx = cmd.index(flag)
            cmd[idx + 1] = "4" if flag == "--epochs" else "2"

    if resume:
        if last_pt.exists():
            print(f"[train] Resuming from {last_pt}  (epoch {_peek_epoch(last_pt)})")
            cmd += ["--resume", str(last_pt)]
        else:
            print(f"[train] --resume requested but {last_pt} not found — starting fresh.")

    print(f"\n[train] Working dir : {ocr_dir}")
    print(f"[train] Command     :\n  " + " ".join(str(c) for c in cmd) + "\n")

    result = subprocess.run(cmd, cwd=str(ocr_dir))
    sys.exit(result.returncode)


def _peek_epoch(path: Path) -> int:
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        return ckpt.get("epoch", "?")
    except Exception:
        return "?"


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Colab session setup + char_classifier training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--resume",        action="store_true",
                   help="Resume from last.pt stored in CKPT_DIR on Drive")
    p.add_argument("--smoke-test",    action="store_true",
                   help="Quick sanity check: 10 img/class, 4 epochs")
    p.add_argument("--skip-clone",    action="store_true",
                   help="Skip git clone/pull (repo already set up this session)")
    p.add_argument("--skip-dataset",  action="store_true",
                   help="Skip dataset sync (already copied to VM this session)")
    p.add_argument("--skip-deps",     action="store_true",
                   help="Skip pip install step")
    p.add_argument("--setup-only",    action="store_true",
                   help="Run setup steps only, do not launch training")
    p.add_argument("--zip-dataset",   action="store_true",
                   help="Zip char-dataset for Drive upload (run locally, then exit)")
    p.add_argument("--zip-output",    default=None,
                   help="Output path for --zip-dataset (default: <repo-root>/char-dataset.zip)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" Colab Training Setup")
    print(f"  Scripts  : {SCRIPTS}")
    print(f"  Epochs   : {EPOCHS}  (freeze={FREEZE_EPOCHS})")
    print(f"  Backbone : {BACKBONE}")
    print(f"  Ckpt dir : {CKPT_DIR}")
    print("=" * 60)

    if args.zip_dataset:
        zip_dataset(output_path=args.zip_output, scripts=SCRIPTS)
        return

    mount_drive()

    if not args.skip_clone:
        clone_or_update_repo()

    if not args.skip_deps:
        install_deps()

    if not args.skip_dataset:
        sync_dataset()

    if args.setup_only:
        print("\n[setup] Setup complete. Run with --resume (or without) to start training.")
        return

    train(resume=args.resume, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
