"""Train an authorship attribution classifier for a guild.

Uses TF-IDF (word + char n-grams) with Logistic Regression — fast, CPU-only,
effective at hundreds of messages per user.

Usage:
    python train.py --guild GUILD_ID [--min-tokens 3] [--min-messages 20] [--split 0.85]
    python train.py --guild GUILD_ID --eval-only   # score existing model without retraining
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

_DATA_ROOT  = Path(__file__).parent.parent / "data"
_MODEL_ROOT = Path.home() / ".tl-bot" / "authorship"

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    from scipy.sparse import hstack
    import numpy as np
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


def _load_split(guild_id: str, split: str) -> tuple[list[str], list[int], list[str]]:
    """Load train.jsonl or val.jsonl. Returns (texts, labels, usernames)."""
    p = _DATA_ROOT / guild_id / f"{split}.jsonl"
    texts, labels, names = [], [], []
    if not p.exists():
        return texts, labels, names
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                texts.append(r["text"])
                labels.append(r["label"])
                names.append(r.get("username", str(r["label"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return texts, labels, names


def _label_names(guild_id: str) -> dict[int, str]:
    """Return {label_int: username} from label_map.json."""
    p = _DATA_ROOT / guild_id / "label_map.json"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        lm = json.load(f)
    return {v["label"]: v["username"] for v in lm.values()}


def train(guild_id: str) -> dict:
    """Train and save the model. Returns eval metrics dict."""
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    train_texts, train_labels, _ = _load_split(guild_id, "train")
    val_texts,   val_labels,   _ = _load_split(guild_id, "val")

    if not train_texts:
        raise ValueError(f"No training data found for guild {guild_id}. Run dataset.py first.")
    if len(set(train_labels)) < 2:
        raise ValueError("Need at least 2 users to train a classifier.")

    # Word unigrams + bigrams capture phrasing; char 3-5-grams capture punctuation & spelling style.
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=20_000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=30_000,
    )

    X_train_w = word_vec.fit_transform(train_texts)
    X_train_c = char_vec.fit_transform(train_texts)
    X_train   = hstack([X_train_w, X_train_c])

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, train_labels)

    # Evaluate
    X_val_w = word_vec.transform(val_texts)
    X_val_c = char_vec.transform(val_texts)
    X_val   = hstack([X_val_w, X_val_c])

    val_preds = clf.predict(X_val)
    acc       = accuracy_score(val_labels, val_preds)
    names     = _label_names(guild_id)
    target_names = [names.get(i, str(i)) for i in sorted(set(val_labels))]
    report    = classification_report(val_labels, val_preds, target_names=target_names)

    # Save artifacts
    out_dir = _MODEL_ROOT / guild_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "word_vec.pkl").open("wb") as f:
        pickle.dump(word_vec, f)
    with (out_dir / "char_vec.pkl").open("wb") as f:
        pickle.dump(char_vec, f)
    with (out_dir / "clf.pkl").open("wb") as f:
        pickle.dump(clf, f)

    # Copy label_map alongside model so identify.py is self-contained
    lm_src = _DATA_ROOT / guild_id / "label_map.json"
    if lm_src.exists():
        (out_dir / "label_map.json").write_text(lm_src.read_text(encoding="utf-8"), encoding="utf-8")

    meta = {
        "guild_id":     guild_id,
        "train_samples": len(train_texts),
        "val_samples":  len(val_texts),
        "users":        len(set(train_labels)),
        "val_accuracy": round(acc, 4),
        "features":     {
            "word_ngrams": "(1,2)",
            "char_ngrams": "(3,5)",
            "word_vocab":  word_vec.vocabulary_.__len__(),
            "char_vocab":  char_vec.vocabulary_.__len__(),
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"accuracy": acc, "report": report, "meta": meta, "model_dir": str(out_dir)}


def evaluate(guild_id: str) -> dict:
    """Score an existing saved model against the current val split."""
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    out_dir = _MODEL_ROOT / guild_id
    for fname in ("word_vec.pkl", "char_vec.pkl", "clf.pkl"):
        if not (out_dir / fname).exists():
            raise FileNotFoundError(f"No trained model found at {out_dir}. Run train first.")

    with (out_dir / "word_vec.pkl").open("rb") as f:
        word_vec = pickle.load(f)
    with (out_dir / "char_vec.pkl").open("rb") as f:
        char_vec = pickle.load(f)
    with (out_dir / "clf.pkl").open("rb") as f:
        clf = pickle.load(f)

    from scipy.sparse import hstack
    val_texts, val_labels, _ = _load_split(guild_id, "val")
    X_val = hstack([word_vec.transform(val_texts), char_vec.transform(val_texts)])
    preds = clf.predict(X_val)

    names = _label_names(guild_id)
    target_names = [names.get(i, str(i)) for i in sorted(set(val_labels))]
    acc    = accuracy_score(val_labels, preds)
    report = classification_report(val_labels, preds, target_names=target_names)
    return {"accuracy": acc, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train authorship attribution model.")
    parser.add_argument("--guild",        required=True, help="Guild ID")
    parser.add_argument("--min-tokens",   type=int, default=3)
    parser.add_argument("--min-messages", type=int, default=20)
    parser.add_argument("--split",        type=float, default=0.85)
    parser.add_argument("--eval-only",    action="store_true", help="Score existing model without retraining")
    args = parser.parse_args()

    if args.eval_only:
        result = evaluate(args.guild)
        print(f"Val accuracy: {result['accuracy']:.4f}\n")
        print(result["report"])
    else:
        # Rebuild dataset then train
        import dataset as _dataset
        print(f"Building dataset for guild {args.guild}...")
        ds_stats = _dataset.build_dataset(
            guild_id=args.guild,
            split=args.split,
            min_tokens=args.min_tokens,
            min_messages=args.min_messages,
        )
        if "error" in ds_stats:
            print(f"Dataset error: {ds_stats['error']}")
            raise SystemExit(1)

        print(f"  {ds_stats['users']} users | {ds_stats['train']} train | {ds_stats['val']} val")
        print("Training...")
        result = train(args.guild)
        print(f"Val accuracy: {result['accuracy']:.4f}")
        print(f"Model saved to: {result['model_dir']}")
        print()
        print(result["report"])
