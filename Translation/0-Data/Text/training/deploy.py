"""Install fine-tuned TL-Bot translation models for local inference.

Copies the best (or latest) checkpoint from training/checkpoints/<direction>/
to ~/.tl-bot/models/<direction>/ so translate_text.py can load it with a
local-first, HF API fallback strategy.

After deploying, translate_text.py automatically prefers the local model —
no code change needed.

Usage:
    python deploy.py --direction mul-en
    python deploy.py --direction en-ko
    python deploy.py --direction mul-en --checkpoint checkpoints/mul-en/latest
    python deploy.py --list           # show installed models
    python deploy.py --remove mul-en  # uninstall a direction
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
MODELS_DIR = Path.home() / ".tl-bot" / "models"


def _installed_directions() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir())


def list_installed() -> None:
    directions = _installed_directions()
    if not directions:
        print(f"No models installed in {MODELS_DIR}")
        return
    print(f"Installed models in {MODELS_DIR}:")
    for d in directions:
        model_dir = MODELS_DIR / d
        config = model_dir / "config.json"
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        print(f"  {d:<12}  {size / 1e6:.1f} MB  {'(config present)' if config.exists() else '(incomplete?)'}")


def deploy(direction: str, checkpoint: Path | None) -> None:
    if checkpoint is None:
        # Prefer best, fall back to latest
        best = CHECKPOINT_DIR / direction / "best"
        latest = CHECKPOINT_DIR / direction / "latest"
        if best.exists():
            checkpoint = best
            print(f"Using best checkpoint: {checkpoint}")
        elif latest.exists():
            checkpoint = latest
            print(f"No best checkpoint found — using latest: {checkpoint}")
        else:
            print(f"No checkpoint found for direction '{direction}'.")
            print(f"Run: python train.py --direction {direction}")
            sys.exit(1)

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    dest = MODELS_DIR / direction
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(checkpoint), str(dest))

    print(f"Installed {direction} → {dest}")
    print(f"\nLoad in translate_text.py with:")
    print(f"  MarianMTModel.from_pretrained('{dest}')")
    print(f"  (translate_text.py will detect and use this automatically)")


def remove(direction: str) -> None:
    dest = MODELS_DIR / direction
    if not dest.exists():
        print(f"Model not installed: {direction}")
        sys.exit(1)
    shutil.rmtree(dest)
    print(f"Removed {direction} from {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy fine-tuned text models for local inference.")
    parser.add_argument("--direction", default=None,
                        help="Model direction to deploy: 'mul-en' or 'en-<lang>'")
    parser.add_argument("--checkpoint", default=None, type=Path,
                        help="Specific checkpoint dir to deploy (default: best, then latest)")
    parser.add_argument("--list", action="store_true", help="List installed models and exit")
    parser.add_argument("--remove", default=None, metavar="DIRECTION",
                        help="Uninstall a deployed model direction")
    args = parser.parse_args()

    if args.list:
        list_installed()
        return

    if args.remove:
        remove(args.remove)
        return

    if not args.direction:
        parser.error("--direction is required (e.g. --direction mul-en)")

    deploy(args.direction, args.checkpoint)


if __name__ == "__main__":
    main()
