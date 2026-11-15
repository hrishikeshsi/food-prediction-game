"""Fit and export the dish decision tree.

    python3 -m ml.decision_tree

Fits scikit-learn's ``DecisionTreeClassifier`` (entropy criterion) on the
question table — one row per dish, one boolean feature per question, the dish
as the class — and writes it out in four forms under ``data/artifacts/tree``:

    decision_tree.joblib   the fitted classifier, with its feature and class names
    decision_tree.txt      an indented outline anyone can read
    decision_tree.dot      Graphviz source (``dot -Tsvg decision_tree.dot -o tree.svg``)
    decision_tree.json     the nested tree the ``/tree`` page draws, plus summary figures

The same tree is fitted in-process by the API (``app.services.tree``), which
walks it for the hint game and serves it at ``/tree``.
"""
from __future__ import annotations

import json
import sys

import joblib

from app.config import settings
from app.services.catalog import get_catalogue
from app.services.tree import canonical_tree


def main() -> int:
    tree = canonical_tree()
    catalogue = get_catalogue()
    out = settings.artifact_dir / "tree"
    out.mkdir(parents=True, exist_ok=True)

    stats = tree.stats()
    joblib.dump(
        {
            "classifier": tree.classifier,
            "feature_keys": tree.table.keys,
            "feature_questions": [q.text for q in catalogue.questions],
            "classes": tree.table.names,
        },
        out / "decision_tree.joblib",
    )
    (out / "decision_tree.txt").write_text(tree.to_text(), encoding="utf-8")
    (out / "decision_tree.dot").write_text(tree.to_dot(), encoding="utf-8")
    (out / "decision_tree.json").write_text(
        json.dumps({"stats": stats, "tree": tree.to_dict()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"DecisionTreeClassifier(criterion='{stats['criterion']}') on "
          f"{stats['dishes']} dishes × {stats['questions_available']} questions")
    print(f"  leaves            {stats['leaves']}  (one per dish)")
    print(f"  depth             mean {stats['mean_depth']:.2f}, max {stats['max_depth']}  "
          f"(ideal log2 = {stats['ideal_depth']:.2f})")
    print(f"  questions used    {stats['questions_used']} of {stats['questions_available']}")
    print(f"  training accuracy {stats['training_accuracy']:.1%}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
