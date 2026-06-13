"""Install the trained TL-Bot OCR model into EasyOCR's model directories.

After running train.py, call this script to make the custom model available
for inference via:

    easyocr.Reader(['en'], recog_network='tl_bot_ocr')

Installs:
    checkpoints/tl_bot_ocr_best.pth  → ~/.EasyOCR/model/tl_bot_ocr.pth
    tl_bot_ocr.yaml                  → ~/.EasyOCR/user_network/tl_bot_ocr.yaml
    tl_bot_ocr.py                    → ~/.EasyOCR/user_network/tl_bot_ocr.py

Usage:
    python deploy.py
    python deploy.py --checkpoint checkpoints/tl_bot_ocr_latest.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
EASYOCR_DIR = Path.home() / ".EasyOCR"
MODEL_DIR = EASYOCR_DIR / "model"
USER_NETWORK_DIR = EASYOCR_DIR / "user_network"

MODEL_NAME = "tl_bot_ocr"


def deploy(checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `python train.py` first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    # Install weights
    dest_pth = MODEL_DIR / f"{MODEL_NAME}.pth"
    shutil.copy2(checkpoint, dest_pth)
    print(f"Installed weights:   {dest_pth}")

    # Install model config files
    for ext in (".yaml", ".py"):
        src = TRAINING_DIR / f"{MODEL_NAME}{ext}"
        if not src.exists():
            print(f"[warn] Missing config file: {src} — skipping")
            continue
        dest = USER_NETWORK_DIR / src.name
        shutil.copy2(src, dest)
        print(f"Installed config:    {dest}")

    print(f"\nDone. Load the model with:")
    print(f"    easyocr.Reader(['en'], recog_network='{MODEL_NAME}')")


def main():
    parser = argparse.ArgumentParser(description="Deploy trained OCR model to EasyOCR.")
    parser.add_argument(
        "--checkpoint",
        default=str(TRAINING_DIR / "checkpoints" / f"{MODEL_NAME}_best.pth"),
        help="Path to the .pth checkpoint to deploy (default: best checkpoint)",
    )
    args = parser.parse_args()
    deploy(Path(args.checkpoint))


if __name__ == "__main__":
    main()
