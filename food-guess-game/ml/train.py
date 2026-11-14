"""Train the TF-IDF + PyTorch dish classifier.

    python3 -m ml.train
    python3 -m ml.train --epochs 40 --max-features 60000

Artifacts written to ``data/artifacts``:
    vectorizer.joblib   fitted scikit-learn feature union
    model.pt            model weights + config
    labels.json         index -> dish name
    metrics.json        held-out accuracy, top-3/top-5, macro-F1, per-class detail
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from torch import nn

from ml.model import DishClassifier, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "artifacts"
SEED = 20260809


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vectorizer(max_features: int) -> FeatureUnion:
    """Word n-grams carry the meaning; character n-grams absorb the typos.

    Players type "biriyani", "bolognaise", "shakshouka" — a word-only model
    treats each misspelling as an unknown token, so a character n-gram view is
    unioned in to keep those queries alive.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                    sublinear_tf=True,
                    max_features=max_features // 2,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def iterate_batches(
    features: sparse.csr_matrix,
    targets: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
):
    order = np.arange(features.shape[0])
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        dense = torch.from_numpy(features[idx].toarray().astype(np.float32)).to(device)
        yield (
            dense,
            torch.from_numpy(targets[idx]).long().to(device),
            torch.from_numpy(weights[idx].astype(np.float32)).to(device),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: sparse.csr_matrix,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    model.eval()
    logits_all = []
    for start in range(0, features.shape[0], batch_size):
        dense = torch.from_numpy(
            features[start : start + batch_size].toarray().astype(np.float32)
        ).to(device)
        logits_all.append(model(dense).cpu())
    logits = torch.cat(logits_all)
    ranked = logits.argsort(dim=1, descending=True).numpy()
    truth = targets

    top1 = float((ranked[:, 0] == truth).mean())
    top3 = float(np.mean([t in r[:3] for r, t in zip(ranked, truth)]))
    top5 = float(np.mean([t in r[:5] for r, t in zip(ranked, truth)]))
    macro_f1 = float(f1_score(truth, ranked[:, 0], average="macro", zero_division=0))
    loss = float(nn.functional.cross_entropy(logits, torch.from_numpy(truth).long()))
    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "macro_f1": macro_f1,
        "loss": loss,
        "predictions": ranked[:, 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ARTIFACT_DIR / "corpus.csv.gz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=8000,
                        help="word n-gram cap; char n-grams get half this")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    set_seed(SEED)
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"corpus not found at {corpus_path} — run ml.prepare_data first")

    frame = pd.read_csv(corpus_path)
    frame = frame.dropna(subset=["text", "label"])
    print(f"corpus: {len(frame):,} rows, {frame['label'].nunique()} labels")

    labels = sorted(frame["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    y = frame["label"].map(label_to_idx).to_numpy()
    w = frame.get("weight", pd.Series(np.ones(len(frame)))).to_numpy(dtype=np.float32)

    x_train_txt, x_val_txt, y_train, y_val, w_train, w_val = train_test_split(
        frame["text"].tolist(), y, w, test_size=args.test_size, random_state=SEED, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    vectorizer = build_vectorizer(args.max_features)
    X_train = vectorizer.fit_transform(x_train_txt).astype(np.float32)
    X_val = vectorizer.transform(x_val_txt).astype(np.float32)
    print(f"features: {X_train.shape[1]:,} ({time.time() - t0:.1f}s to fit)")

    config = ModelConfig(input_dim=X_train.shape[1], num_classes=len(labels))
    model = DishClassifier(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    best_f1 = -1.0
    best_state: dict | None = None
    epochs_without_gain = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb, wb in iterate_batches(
            X_train, y_train, w_train, args.batch_size, True, device
        ):
            optimiser.zero_grad(set_to_none=True)
            loss = (criterion(model(xb), yb) * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += float(loss) * len(yb)
            seen += len(yb)
        scheduler.step()

        val = evaluate(model, X_val, y_val, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_loss": val["loss"],
                "val_top1": val["top1"],
                "val_top3": val["top3"],
                "val_macro_f1": val["macro_f1"],
            }
        )
        print(
            f"epoch {epoch:>3}  train_loss {running / max(seen, 1):.4f}  "
            f"val_loss {val['loss']:.4f}  top1 {val['top1']:.4f}  "
            f"top3 {val['top3']:.4f}  macroF1 {val['macro_f1']:.4f}"
        )

        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"early stopping at epoch {epoch} (no gain for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, X_val, y_val, device)

    # Per-class recall, so it is obvious which dishes the model confuses.
    per_class: dict[str, float] = {}
    preds = final["predictions"]
    for idx, name in enumerate(labels):
        mask = y_val == idx
        if mask.any():
            per_class[name] = float((preds[mask] == idx).mean())
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.joblib")
    torch.save(
        {"state_dict": model.state_dict(), "config": config.to_dict()},
        ARTIFACT_DIR / "model.pt",
    )
    (ARTIFACT_DIR / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))

    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": int(len(frame)),
        "labels": len(labels),
        "features": int(X_train.shape[1]),
        "parameters": int(n_params),
        "epochs_run": len(history),
        "val_top1": final["top1"],
        "val_top3": final["top3"],
        "val_top5": final["top5"],
        "val_macro_f1": final["macro_f1"],
        "worst_classes": [{"dish": n, "recall": r} for n, r in worst],
        "history": history,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== held-out results ===")
    print(f"top-1     {final['top1']:.4f}")
    print(f"top-3     {final['top3']:.4f}")
    print(f"top-5     {final['top5']:.4f}")
    print(f"macro F1  {final['macro_f1']:.4f}")
    print("weakest classes:", ", ".join(f"{n} ({r:.2f})" for n, r in worst[:5]))
    print(f"\nartifacts written to {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
