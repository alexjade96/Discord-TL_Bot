"""Deploy the user recognition model for a guild.

Runs dataset.py → train.py in sequence and installs the resulting artifacts
to ~/.tl-bot/user-recognition/<guild_id>/ so identify.py can load them automatically.

Usage:
    python deploy.py --guild GUILD_ID
    python deploy.py --guild GUILD_ID --split 0.85 --min-tokens 3 --min-messages 20
    python deploy.py --list           # show all installed guild models
    python deploy.py --remove GUILD_ID
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_MODEL_ROOT = Path.home() / ".tl-bot" / "user-recognition"
_TRAINING_DIR = Path(__file__).parent


def list_installed() -> None:
    if not _MODEL_ROOT.exists() or not any(_MODEL_ROOT.iterdir()):
        print(f"No models installed in {_MODEL_ROOT}")
        return
    print(f"Installed user recognition models in {_MODEL_ROOT}:")
    for d in sorted(_MODEL_ROOT.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            users = meta.get("users", "?")
            acc   = meta.get("val_accuracy", "?")
            train = meta.get("train_samples", "?")
            print(f"  {d.name}  |  {users} users  |  {train} train samples  |  val_acc={acc}")
        else:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {d.name}  |  {size / 1e3:.0f} KB")


def remove(guild_id: str) -> None:
    dest = _MODEL_ROOT / guild_id
    if not dest.exists():
        print(f"No model installed for guild {guild_id}")
        sys.exit(1)
    shutil.rmtree(dest)
    print(f"Removed {guild_id} from {_MODEL_ROOT}")


def deploy(guild_id: str, split: float, min_tokens: int, min_messages: int) -> None:
    sys.path.insert(0, str(_TRAINING_DIR))
    import dataset as _dataset
    import train   as _train

    print(f"Building dataset for guild {guild_id}...")
    stats = _dataset.build_dataset(
        guild_id=guild_id,
        split=split,
        min_tokens=min_tokens,
        min_messages=min_messages,
    )
    if "error" in stats:
        print(f"Dataset error: {stats['error']}")
        sys.exit(1)
    print(f"  {stats['users']} users | {stats['train']} train | {stats['val']} val")

    print("Training classifier...")
    result = _train.train(guild_id)
    print(f"  Val accuracy: {result['accuracy']:.4f}")
    print(f"  Model saved:  {result['model_dir']}")
    print()
    print(result["report"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy user recognition model.")
    parser.add_argument("--guild",        default=None, help="Guild ID to deploy")
    parser.add_argument("--split",        type=float, default=0.85, help="Train fraction (default 0.85)")
    parser.add_argument("--min-tokens",   type=int,   default=3,    help="Min tokens per message (default 3)")
    parser.add_argument("--min-messages", type=int,   default=20,   help="Min messages per user (default 20)")
    parser.add_argument("--list",         action="store_true", help="List installed models and exit")
    parser.add_argument("--remove",       default=None, metavar="GUILD_ID", help="Uninstall a guild's model")
    args = parser.parse_args()

    if args.list:
        list_installed()
        return

    if args.remove:
        remove(args.remove)
        return

    if not args.guild:
        parser.error("--guild is required (e.g. --guild 1502045408677986405)")

    deploy(args.guild, args.split, args.min_tokens, args.min_messages)


if __name__ == "__main__":
    main()
