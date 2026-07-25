"""Authorship attribution inference.

Two backends share one public API. Which one serves a guild is decided by the
"model_type" field in ~/.tl-bot/authorship/<guild_id>/meta.json:

    tfidf   TF-IDF + Logistic Regression, installed by
            UserRecognition/0-Data/training/deploy.py (the shipped baseline)
    neural  fine-tuned transformer, installed by
            Models/UserRecognition/author_classifier/deploy.py

When both are installed the neural model wins; removing it falls back to
TF-IDF. Callers see no difference.

Public API:
    identify(text, guild_id) -> list[dict]   # sorted by score descending
    model_exists(guild_id)   -> bool
    model_type(guild_id)     -> str | None   # 'tfidf' | 'neural' | None

CLI:
    python identify.py --guild GUILD_ID --text "some message text"
    python identify.py --guild GUILD_ID --text "some message text" --top 3
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

try:
    from scipy.sparse import hstack as _hstack
except ImportError:  # graceful import failure; tests mock this
    _hstack = None  # type: ignore[assignment]

_MODEL_ROOT = Path.home() / ".tl-bot" / "authorship"

_TFIDF_FILES  = ("word_vec.pkl", "char_vec.pkl", "clf.pkl")
_NEURAL_FILES = ("model.pt", "config.json", "class_names.json")

_TRAIN_HINT = (
    "Run: python UserRecognition/0-Data/training/deploy.py --guild {gid}\n"
    "  or: python -m author_classifier.train --guild {gid}   (from Models/UserRecognition/)"
)

# Module-level cache: {guild_id: (backend, payload)}
_cache: dict = {}


def _has(d: Path, files) -> bool:
    return all((d / f).exists() for f in files)


def model_type(guild_id: str) -> str | None:
    """Return which backend serves this guild, or None if nothing is installed."""
    d = _MODEL_ROOT / str(guild_id)
    if _has(d, _NEURAL_FILES):
        return "neural"
    if _has(d, _TFIDF_FILES):
        return "tfidf"
    return None


def model_exists(guild_id: str) -> bool:
    return model_type(guild_id) is not None


def _load_label_map(d: Path) -> dict[int, str]:
    lm_path = d / "label_map.json"
    if not lm_path.exists():
        return {}
    raw = json.loads(lm_path.read_text(encoding="utf-8"))
    return {v["label"]: v["username"] for v in raw.values()}


def _load_tfidf(d: Path):
    with (d / "word_vec.pkl").open("rb") as f:
        word_vec = pickle.load(f)
    with (d / "char_vec.pkl").open("rb") as f:
        char_vec = pickle.load(f)
    with (d / "clf.pkl").open("rb") as f:
        clf = pickle.load(f)
    return word_vec, char_vec, clf, _load_label_map(d)


def _load_neural(d: Path):
    import sys

    import torch

    # author_classifier lives in the research tree; add it to the path only when
    # a neural model is actually being served, so the TF-IDF path stays free of
    # any torch/transformers import.
    ac_dir = Path(__file__).parent.parent / "Models" / "UserRecognition"
    if str(ac_dir) not in sys.path:
        sys.path.insert(0, str(ac_dir))
    from author_classifier.model_builder import create_model, create_tokenizer  # noqa: E402

    config      = json.loads((d / "config.json").read_text(encoding="utf-8"))
    class_names = json.loads((d / "class_names.json").read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(d / "model.pt", map_location=device, weights_only=False)

    model = create_model(config["backbone"], num_classes=len(class_names), freeze_base=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tokenizer = create_tokenizer(config["backbone"])

    return model, tokenizer, class_names, config.get("max_length", 256), device


def _load(guild_id: str):
    if guild_id in _cache:
        return _cache[guild_id]

    d = _MODEL_ROOT / str(guild_id)
    backend = model_type(guild_id)
    if backend is None:
        raise FileNotFoundError(
            f"No trained model for guild {guild_id}. "
            + _TRAIN_HINT.format(gid=guild_id)
        )

    payload = _load_neural(d) if backend == "neural" else _load_tfidf(d)
    _cache[guild_id] = (backend, payload)
    return _cache[guild_id]


def _identify_tfidf(text: str, payload) -> list[dict]:
    word_vec, char_vec, clf, label_map = payload
    X = _hstack([word_vec.transform([text]), char_vec.transform([text])])
    probs = clf.predict_proba(X)[0]
    return [
        {
            "username": label_map.get(int(cls), str(cls)),
            "label":    int(cls),
            "score":    round(float(prob), 4),
        }
        for cls, prob in zip(clf.classes_, probs)
    ]


def _identify_neural(text: str, payload) -> list[dict]:
    import torch

    model, tokenizer, class_names, max_length, device = payload
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**enc)[0], dim=-1).cpu().tolist()
    return [
        {
            "username": class_names[i] if i < len(class_names) else str(i),
            "label":    i,
            "score":    round(float(p), 4),
        }
        for i, p in enumerate(probs)
    ]


def identify(text: str, guild_id: str) -> list[dict]:
    """
    Return ranked list of probable authors for text.

    Each entry: {"username": str, "label": int, "score": float}
    Sorted by score descending (most likely first).
    """
    backend, payload = _load(str(guild_id))
    results = (_identify_neural if backend == "neural" else _identify_tfidf)(text, payload)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rank likely authors of a message.")
    parser.add_argument("--guild", required=True, help="Guild ID")
    parser.add_argument("--text",  required=True, help="Message text to classify")
    parser.add_argument("--top",   type=int, default=0, help="Show top N results (default: all)")
    args = parser.parse_args()

    if not model_exists(args.guild):
        print(f"No trained model for guild {args.guild}.")
        print(_TRAIN_HINT.format(gid=args.guild))
        raise SystemExit(1)

    results = identify(args.text, args.guild)
    if args.top:
        results = results[: args.top]

    print(f"Backend: {model_type(args.guild)}")
    print(f"Text: {args.text!r}\n")
    for rank, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"  {rank}. {r['username']:<20} {bar}  {r['score']*100:.1f}%")
