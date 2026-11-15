"""The fitted decision tree: scikit-learn's ``DecisionTreeClassifier`` over the bank.

The classifier is trained on the same table the Bayesian engine uses — one row
per dish, one boolean column per question, the dish as the class label — with
the entropy criterion, so every node holds the question that best halves the
dishes still in play. Because no two dishes share an answer pattern, the tree
grows until each leaf is exactly one dish.

Two things use it:

* **Reverse mode.** The hints for a hidden dish are that dish's root-to-leaf
  path, read out as statements: broad, even splits near the root, the
  distinguishing detail near the leaf. A wrong guess is answered with the split
  where the guess and the hidden dish part ways.
* **Inspection.** The canonical tree is exported and served at ``/tree``, and
  the notebook uses it as the hard-split baseline the Bayesian engine is
  measured against.

The canonical tree (index 0) considers every question at every node. The
others consider a random subset, so they are slightly deeper but open with
different questions — which keeps successive rounds from starting identically.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from app.config import settings
from app.services.catalog import dish_emoji, get_catalogue
from app.services.engine import QuestionTable, question_table

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    """One node on a dish's path: the question asked there and its answer."""

    key: str
    truth: bool
    dishes_before: int   # dishes reaching this node
    dishes_after: int    # dishes continuing down the dish's branch


class DishTree:
    """A fitted classifier plus the per-dish paths read off it."""

    def __init__(self, classifier: DecisionTreeClassifier, table: QuestionTable) -> None:
        self.classifier = classifier
        self.table = table
        tree = classifier.tree_
        self._feature = tree.feature
        self._no = tree.children_left      # feature <= 0.5, i.e. the answer is no
        self._yes = tree.children_right
        self._samples = tree.n_node_samples
        self._label = tree.value.reshape(tree.node_count, -1).argmax(axis=1)

        self._paths: dict[str, list[Split]] = {}
        for column, name in enumerate(table.names):
            node, path = 0, []
            while self._no[node] != -1:
                question = int(self._feature[node])
                truth = bool(table.truth[question, column])
                child = int(self._yes[node] if truth else self._no[node])
                path.append(Split(table.keys[question], truth,
                                  int(self._samples[node]), int(self._samples[child])))
                node = child
            if self._samples[node] != 1 or self._label[node] != column:
                raise RuntimeError(
                    f"decision tree does not isolate {name!r}; two dishes must share "
                    "an answer to every question"
                )
            self._paths[name] = path

    # -- per-dish paths -----------------------------------------------------
    def path(self, dish: str) -> list[Split]:
        return self._paths[dish]

    def depth(self, dish: str) -> int:
        return len(self._paths[dish])

    def classify(self, answer) -> tuple[str, int]:
        """Walk the tree with ``answer(question_key) -> bool``; (dish reached, questions used)."""
        node, used = 0, 0
        while self._no[node] != -1:
            key = self.table.keys[int(self._feature[node])]
            node = int(self._yes[node] if answer(key) else self._no[node])
            used += 1
        return self.table.names[int(self._label[node])], used

    # -- summaries ----------------------------------------------------------
    def stats(self) -> dict:
        depths = np.array([len(p) for p in self._paths.values()])
        predicted = self.classifier.predict(self.table.truth.T.astype(np.uint8))
        return {
            "dishes": len(self.table.names),
            "questions_available": len(self.table.keys),
            "questions_used": int(len({s.key for p in self._paths.values() for s in p})),
            "nodes": int(self.classifier.tree_.node_count),
            "leaves": int(self.classifier.get_n_leaves()),
            "max_depth": int(depths.max()),
            "mean_depth": float(depths.mean()),
            "ideal_depth": float(np.log2(len(depths))),
            "training_accuracy": float((predicted == np.arange(len(predicted))).mean()),
            "criterion": self.classifier.criterion,
        }

    def to_dict(self) -> dict:
        """The whole tree as nested JSON, for the viewer."""
        catalogue = get_catalogue()

        def build(node: int) -> dict:
            if self._no[node] == -1:
                name = self.table.names[int(self._label[node])]
                return {"dish": name, "emoji": dish_emoji(catalogue.get(name))}
            question = catalogue.questions[int(self._feature[node])]
            return {
                "key": question.key,
                "question": question.text,
                "dishes": int(self._samples[node]),
                "yes": build(int(self._yes[node])),
                "no": build(int(self._no[node])),
            }

        return build(0)

    def to_text(self) -> str:
        """The whole tree as an indented outline."""
        catalogue = get_catalogue()
        lines: list[str] = []

        def walk(node: int, prefix: str, label: str) -> None:
            if self._no[node] == -1:
                lines.append(f"{prefix}{label}→ {self.table.names[int(self._label[node])]}")
                return
            question = catalogue.questions[int(self._feature[node])]
            lines.append(f"{prefix}{label}{question.text}  [{int(self._samples[node])} dishes]")
            walk(int(self._yes[node]), prefix + "    ", "yes: ")
            walk(int(self._no[node]), prefix + "    ", "no:  ")

        walk(0, "", "")
        return "\n".join(lines) + "\n"

    def to_dot(self) -> str:
        """Graphviz source, as scikit-learn renders it."""
        catalogue = get_catalogue()
        return export_graphviz(
            self.classifier,
            feature_names=[q.text for q in catalogue.questions],
            class_names=self.table.names,
            filled=False, rounded=True, impurity=False, proportion=False,
        )


def _fit(max_features: float | None, seed: int) -> DecisionTreeClassifier:
    table = question_table()
    features = table.truth.T.astype(np.uint8)          # dishes × questions
    labels = np.arange(len(table.names))               # every dish is its own class
    classifier = DecisionTreeClassifier(
        criterion="entropy", max_features=max_features, random_state=seed
    )
    with warnings.catch_warnings():
        # scikit-learn warns that one sample per class looks like regression.
        # Here it is the point: the tree's job is to tell every dish apart.
        warnings.simplefilter("ignore", UserWarning)
        classifier.fit(features, labels)
    return classifier


def _fingerprint() -> str:
    """Changes whenever a cached fit would no longer describe the catalogue."""
    table = question_table()
    digest = hashlib.sha256()
    digest.update(table.truth.tobytes())
    digest.update("\n".join(table.keys + table.names).encode("utf-8"))
    digest.update(
        f"{sklearn.__version__}|{settings.hint_trees}|{settings.hint_tree_max_features}".encode()
    )
    return digest.hexdigest()


def _classifiers() -> list[DecisionTreeClassifier]:
    """Fit the trees, or reuse an earlier fit of exactly this table.

    Fitting takes about a second per tree — 350 classes make every candidate
    split expensive — so the result is kept beside the other model artifacts.
    The fingerprint is part of the file name, so a fit made from a different
    catalogue or by a different scikit-learn is never even opened.
    """
    cache = settings.artifact_dir / f"tree_cache_{_fingerprint()[:16]}.joblib"
    if cache.exists():
        try:
            return joblib.load(cache)
        except Exception as exc:  # an unreadable cache is just a miss
            log.warning("ignoring unreadable tree cache: %s", exc)

    classifiers = [_fit(None, 0)]
    for seed in range(1, max(1, settings.hint_trees)):
        classifiers.append(_fit(settings.hint_tree_max_features, seed))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifiers, cache)
        # Keep the few most recent fits (say, one per Python environment in
        # use) and drop the rest, so edits to the catalogue do not pile up.
        stale = sorted(cache.parent.glob("tree_cache_*.joblib"),
                       key=lambda path: path.stat().st_mtime, reverse=True)[3:]
        for path in stale:
            path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not write tree cache: %s", exc)
    return classifiers


@lru_cache(maxsize=1)
def forest() -> tuple[DishTree, ...]:
    """Tree 0 is canonical; the rest are randomised variants for hint variety."""
    table = question_table()
    return tuple(DishTree(classifier, table) for classifier in _classifiers())


def canonical_tree() -> DishTree:
    return forest()[0]
