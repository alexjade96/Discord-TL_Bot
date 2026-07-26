"""End-to-end demo for the UserRecognition user recognition pipeline.

Loads collected message data, builds train/val splits, trains the classifier,
and runs identify() on sample texts — verifying the full pipeline without the bot.

Usage:
    python demo.py --guild GUILD_ID
    python demo.py --guild GUILD_ID --no-train   # skip retraining, use existing model
    python demo.py --guild GUILD_ID --top 3      # show top-3 per sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).parent.parent / "training"
_INFERENCE_DIR = Path(__file__).parent.parent.parent  # Models/UserRecognition/

sys.path.insert(0, str(_TRAINING_DIR))
sys.path.insert(0, str(_INFERENCE_DIR))

import collect_history
import dataset as _dataset
import train   as _train
import identify as _identify

_SAMPLE_TEXTS = [
    "hello how are you doing",
    "can you translate this for me please",
    "what does this say",
    "good morning everyone",
    "thanks for the help",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="UserRecognition pipeline demo.")
    parser.add_argument("--guild",    required=True, help="Guild ID")
    parser.add_argument("--no-train", action="store_true", help="Skip retraining; use existing model")
    parser.add_argument("--top",      type=int, default=0, help="Show top-N results per sample (default: all)")
    args = parser.parse_args()

    guild_id = args.guild

    # ── Dataset stats ────────────────────────────────────────────────────────
    stats = collect_history.dataset_stats(guild_id)
    if guild_id not in stats:
        print(f"No data found for guild {guild_id}.")
        print("Collect first: @TL-Bot /collect <username>")
        sys.exit(1)

    s = stats[guild_id]
    print(f"Guild {guild_id}: {s['messages']} messages from {s['users']} user(s) across {s['channels']} channel(s)")

    # ── Build dataset ────────────────────────────────────────────────────────
    if not args.no_train:
        print("\nBuilding dataset...")
        ds = _dataset.build_dataset(guild_id=guild_id, split=0.85, min_tokens=3, min_messages=20)
        if "error" in ds:
            print(f"Dataset error: {ds['error']}")
            sys.exit(1)
        print(f"  {ds['users']} users | {ds['train']} train | {ds['val']} val")
        for uid, info in ds["label_map"].items():
            print(f"  [{info['label']}] {info['username']} ({uid})")

        # ── Train ────────────────────────────────────────────────────────────
        print("\nTraining classifier...")
        result = _train.train(guild_id)
        print(f"  Val accuracy: {result['accuracy']:.4f}")
        print(result["report"])
    else:
        if not _identify.model_exists(guild_id):
            print(f"No trained model found for guild {guild_id}. Run without --no-train first.")
            sys.exit(1)
        print("(Skipping training — using existing model)\n")

    # ── Identify sample texts ────────────────────────────────────────────────
    # Invalidate cache so a freshly trained model is used
    _identify._cache.pop(guild_id, None)

    print("Running identify() on sample texts:")
    for text in _SAMPLE_TEXTS:
        results = _identify.identify(text, guild_id)
        if args.top:
            results = results[: args.top]
        top = results[0]
        # ASCII bar/arrow: the Windows console defaults to cp1252, which cannot
        # encode block-drawing characters or arrows.
        filled = int(top["score"] * 20)
        bar = "#" * filled + "." * (20 - filled)
        others = "  ".join(f"{r['username']}={r['score']*100:.0f}%" for r in results[1:])
        print(f"  {text!r}")
        print(f"    -> {top['username']:<20} {bar}  {top['score']*100:.1f}%  [{others}]")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
