#!/usr/bin/env python3
"""
remote_train.py -- Remote/cloud session setup + training launcher for char_classifier.

Usage (VS Code tunnel terminal or Colab shell cell):

    # First session — full setup then train
    python Models/remote_train.py

    # Subsequent sessions — skip clone/dataset sync, resume from last.pt
    python Models/remote_train.py --resume

    # Skip individual setup steps if already done this session
    python Models/remote_train.py --resume --skip-clone --skip-dataset

    # Smoke test (10 images/class, 4 epochs) before committing to a full run
    python Models/remote_train.py --smoke-test

Bootstrap (paste into a fresh Colab terminal before the repo is cloned):

    git clone https://github.com/alexjade96/Discord-TL_Bot /content/Discord-TL_Bot
    python /content/Discord-TL_Bot/Models/remote_train.py
"""

import argparse
import os
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

# Where the repo is cloned on the remote VM (fast local SSD).
REPO_DIR = "/content/Discord-TL_Bot"

# Path to a zipped copy of char-dataset on Drive.
# Upload char-dataset.zip to DRIVE_ROOT/ before the first session.
# Alternatively set to None to copy from DRIVE_ROOT/char-dataset/ directly.
DATASET_ZIP = f"{DRIVE_ROOT}/char-dataset.zip"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

SCRIPTS       = ["latin"]        # latin | kana | hangul | cjk | all
EPOCHS        = 48

# CKPT_DIR is derived from SCRIPTS; use _make_ckpt_dir() whenever SCRIPTS may
# have been overridden by CLI args (see main()).
_ALL_SCRIPTS = {"latin", "kana", "hangul", "cjk"}


def _make_ckpt_dir(scripts: list) -> str:
    """Return the checkpoint directory for the given script list.

    - All four scripts (or 'all')  -> checkpoints/
    - Single script                -> checkpoints/<script>/
    - Subset of scripts            -> checkpoints/<a>_<b>_.../ (sorted)
    """
    s = _ALL_SCRIPTS if "all" in scripts else set(scripts)
    if s >= _ALL_SCRIPTS:
        return f"{DRIVE_ROOT}/checkpoints"
    if len(scripts) == 1:
        return f"{DRIVE_ROOT}/checkpoints/{scripts[0]}"
    return f"{DRIVE_ROOT}/checkpoints/{'_'.join(sorted(s))}"


CKPT_DIR = _make_ckpt_dir(SCRIPTS)
FREEZE_EPOCHS   = 3                # head-only warm-up epochs before backbone fine-tune
UNFREEZE_BLOCKS = 4
BATCH_SIZE      = 64
BACKBONE        = "dinov2_vits14"  # dinov2_vits14 | dinov2_vitb14 | convnext_tiny
GRID_MODE       = "all"            # single | rotated | all
MIXUP_ALPHA     = 0.2              # 0.4 caused persistent train<val gap; 0.2 is gentler
SCHEDULER       = "cosine"         # cosine | cosine-warm | none
CLIP_GRAD       = 1.0
LR              = 1e-3             # head LR; backbone uses LR * 0.1


# ============================================================
# HELPERS
# ============================================================

def _run(cmd: str, cwd: str = None, check: bool = True):
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        # Name the failing step. Without this the caller only ever saw the
        # launcher's own exit 1, with no indication of which command produced it.
        print(f"[error] Command failed (exit {result.returncode}): {cmd}", flush=True)
        if check:
            sys.exit(result.returncode)
    return result.returncode


def _in_colab() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return False
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
        print("[setup] Not in Colab - skipping Drive mount.")
        return
    from google.colab import drive
    drive.mount("/content/drive")


def clone_or_update_repo():
    if (Path(REPO_DIR) / ".git").exists():
        print(f"[setup] Repo exists at {REPO_DIR} - pulling latest.")
        _run(f"git -C {REPO_DIR} pull --ff-only")
    else:
        print(f"[setup] Cloning {REPO_URL} -> {REPO_DIR}")
        if GITHUB_USERNAME == "YOUR_USERNAME":
            print("[setup] ERROR: Set GITHUB_USERNAME in remote_train.py before running.")
            sys.exit(1)
        _run(f"git clone {REPO_URL} {REPO_DIR}")


def install_deps():
    print("[setup] Checking / installing packages ...")
    # torch, torchvision, numpy, PIL, sklearn, tqdm are pre-installed on Colab.
    # wordninja/lingua belong to the translation pipeline, not char_classifier —
    # nothing under Models/OCR/ imports them. Non-fatal so a transient PyPI or
    # resolver failure cannot abort a training run that does not need them.
    if _run("pip install -q wordninja lingua-language-detector", check=False) != 0:
        print("[setup] Warning: optional dep install failed - continuing "
              "(char_classifier does not import these).")


def sync_dataset():
    """
    Copy char-dataset from Drive to fast VM-local SSD.
    Skipped if all required script subdirs already exist locally and are populated.
    Prefers DATASET_ZIP; falls back to a plain directory at DRIVE_ROOT/char-dataset/.
    """
    local_root = Path(REPO_DIR) / "Models" / "Datasets" / "char-dataset"
    scripts_needed = (
        {"latin", "kana", "hangul", "cjk"} if "all" in SCRIPTS else set(SCRIPTS)
    )

    def _populated(script: str) -> bool:
        # Existence alone is not enough: an extraction killed partway leaves the
        # dir behind, and a bare is_dir() check would let the next run train on
        # a fraction of the data without saying so.
        d = local_root / script
        return d.is_dir() and any(d.iterdir())

    if all(_populated(s) for s in scripts_needed):
        print(f"[setup] Dataset already present at {local_root} - skipping sync.")
        return

    zip_path = Path(DATASET_ZIP) if DATASET_ZIP else None
    if zip_path and zip_path.exists():
        print(f"[setup] Extracting {zip_path} -> {local_root.parent} ...")
        local_root.parent.mkdir(parents=True, exist_ok=True)
        # Extracted with Python's zipfile, not the unzip binary. zip_dataset()
        # writes this archive, and entry names carry non-ASCII font names (Korean
        # "맑은 고딕" and similar). unzip compares each entry's local-header name
        # against the central-directory name, reports a mismatch for every one of
        # them, and exits 1 — after extracting the file correctly. That exit code
        # aborted the first run of every session; the retry then skipped this step
        # because the script dirs already existed, which is why it "worked the
        # second time". zipfile reads the central directory only, so the whole
        # warning class disappears. It also drops the dependency on an external
        # unzip binary, matching how cell 2 already extracts on Kaggle.
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            for i, name in enumerate(names, 1):
                zf.extract(name, local_root.parent)
                if i % 10000 == 0:
                    print(f"  {i}/{len(names)} files ...", flush=True)
        print(f"[setup] Extracted {len(names)} files.")

        missing = sorted(s for s in scripts_needed if not _populated(s))
        if missing:
            print(f"\n[setup] ERROR: extraction finished but these script dirs "
                  f"are missing under {local_root}: {missing}")
            sys.exit(1)
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
# DATASET ZIP (run locally before first remote session)
# ============================================================

def zip_dataset(output_path: str = None, scripts: list = None):
    """
    Zip Models/Datasets/char-dataset/ for upload to Drive.

    The archive always contains a top-level 'char-dataset/' folder so that
    sync_dataset()'s 'unzip -d <Datasets/>'' unpacks to the correct location.

    Run locally (Windows):
        python Models/remote_train.py --zip-dataset
        python Models/remote_train.py --zip-dataset --scripts latin kana
        python Models/remote_train.py --zip-dataset --zip-output D:/upload/char-dataset.zip

    Then upload the resulting zip to:
        My Drive/Colab Notebooks/TL-Bot/char-dataset.zip
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
    print(f"\n  Upload to Drive: My Drive/Colab Notebooks/TL-Bot/char-dataset.zip")


# ============================================================
# TRAINING LAUNCHER
# ============================================================

def _last_pt_path() -> Path:
    # train.py only auto-scopes into <script>/ when using its default checkpoint
    # dir. We always pass --checkpoint-dir explicitly, so it writes flat to CKPT_DIR.
    return Path(CKPT_DIR) / "last.pt"


def _rclone_sync(src: str, dst: str):
    print(f"\n[sync] {src} → {dst}")
    r = subprocess.run(["rclone", "sync", src, dst])
    if r.returncode != 0:
        print(f"[sync] Warning: rclone returned {r.returncode}")
    else:
        print("[sync] OK")


def _sync_loop(proc, src: str, dst: str, interval: int = 600):
    """Sync checkpoints every `interval` seconds while training subprocess runs."""
    import time
    print(f"[sync] Auto-sync every {interval}s: {src} → {dst}")
    last = 0.0
    while proc.poll() is None:
        if time.time() - last >= interval:
            _rclone_sync(src, dst)
            last = time.time()
        time.sleep(10)
    _rclone_sync(src, dst)
    print("[sync] Final sync complete.")


def train(resume: bool, smoke_test: bool, sync_to: str = None):
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
        "--scheduler",      SCHEDULER,
        "--clip-grad",      str(CLIP_GRAD),
        "--lr",             str(LR),
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
            _print_checkpoint_info(last_pt)
            cmd += ["--resume", str(last_pt)]
        else:
            print(f"[train] --resume requested but {last_pt} not found - starting fresh.")

    print(f"\n[train] Working dir : {ocr_dir}")
    print(f"[train] Command     :\n  " + " ".join(str(c) for c in cmd) + "\n")

    if sync_to:
        sync_src = str(Path(DRIVE_ROOT) / "checkpoints")
        proc = subprocess.Popen(cmd, cwd=str(ocr_dir))
        _sync_loop(proc, sync_src, sync_to)
        sys.exit(proc.returncode)
    else:
        result = subprocess.run(cmd, cwd=str(ocr_dir))
        sys.exit(result.returncode)


def _peek_epoch(path: Path) -> int:
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        return ckpt.get("epoch", "?")
    except Exception:
        return "?"


def _print_checkpoint_info(path: Path):
    try:
        import torch
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        vacc  = ckpt.get("val_acc", 0.0)
        meta  = ckpt.get("meta", {})
        print(f"[train] Resuming from {path}")
        print(f"  Epoch      : {epoch} / {meta.get('total_epochs', '?')}"
              f"  ({meta.get('epochs_remaining', '?')} remaining)")
        print(f"  Phase      : {meta.get('phase', '?')} - {meta.get('phase_label', '')}")
        print(f"  Val acc    : {vacc:.4f}  (best: {meta.get('best_val_acc', vacc):.4f})")
        print(f"  Scripts    : {meta.get('scripts', '?')}")
        print(f"  Backbone   : {meta.get('backbone', '?')}")
        print(f"  Saved at   : {meta.get('saved_at', 'unknown')}")
    except Exception:
        print(f"[train] Resuming from {path}  (epoch {_peek_epoch(path)})")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Remote/cloud session setup + char_classifier training launcher",
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
    p.add_argument("--scripts",        nargs="+", default=None, metavar="SCRIPT",
                   help="Override SCRIPTS config (e.g. --scripts latin kana)")
    p.add_argument("--epochs",         type=int, default=None,
                   help="Override EPOCHS config")
    p.add_argument("--freeze-epochs",  type=int, default=None,
                   help="Override FREEZE_EPOCHS (head warm-up epochs; default 3)")
    p.add_argument("--mixup-alpha",    type=float, default=None,
                   help="Override MIXUP_ALPHA (default 0.2)")
    p.add_argument("--scheduler",      default=None,
                   choices=["cosine", "cosine-warm", "none"],
                   help="Override SCHEDULER (default cosine)")
    p.add_argument("--lr",             type=float, default=None,
                   help="Override LR (head learning rate; backbone = LR * 0.1; default 1e-3)")
    p.add_argument("--storage-root",  default=None,
                   help="Override DRIVE_ROOT for checkpoints and dataset zip "
                        "(Lightning AI: /teamspace/studios/this_studio/TL-Bot)")
    p.add_argument("--repo-dir",      default=None,
                   help="Override REPO_DIR (local runs: path to the cloned repo root)")
    p.add_argument("--sync-to",       default=None,
                   help="rclone destination for automatic checkpoint sync during training "
                        "(e.g. 'gdrive:Colab Notebooks/TL-Bot/checkpoints/'). "
                        "Syncs every 10 minutes and once on finish. Kaggle use only.")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides before anything reads these globals.
    global SCRIPTS, EPOCHS, CKPT_DIR, DRIVE_ROOT, DATASET_ZIP, REPO_DIR
    global FREEZE_EPOCHS, MIXUP_ALPHA, SCHEDULER, LR
    if args.storage_root is not None:
        DRIVE_ROOT  = args.storage_root
        DATASET_ZIP = f"{DRIVE_ROOT}/char-dataset.zip"
    if args.repo_dir is not None:
        REPO_DIR = args.repo_dir
    if args.scripts is not None:
        SCRIPTS = args.scripts
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.freeze_epochs is not None:
        FREEZE_EPOCHS = args.freeze_epochs
    if args.mixup_alpha is not None:
        MIXUP_ALPHA = args.mixup_alpha
    if args.scheduler is not None:
        SCHEDULER = args.scheduler
    if args.lr is not None:
        LR = args.lr
    CKPT_DIR = _make_ckpt_dir(SCRIPTS)

    print("=" * 60)
    print(" Remote Training Setup")
    print(f"  Scripts   : {SCRIPTS}")
    print(f"  Epochs    : {EPOCHS}  (freeze={FREEZE_EPOCHS})")
    print(f"  Backbone  : {BACKBONE}")
    print(f"  Scheduler : {SCHEDULER}  mixup={MIXUP_ALPHA}")
    print(f"  Ckpt dir  : {CKPT_DIR}")
    print("=" * 60)

    if args.zip_dataset:
        zip_dataset(output_path=args.zip_output, scripts=args.scripts)
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

    train(resume=args.resume, smoke_test=args.smoke_test, sync_to=args.sync_to)


if __name__ == "__main__":
    main()
