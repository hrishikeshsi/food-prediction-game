"""Serving wrapper around the trained TF-IDF + PyTorch classifier.

The model is loaded once at process start and held in memory. If the artifacts
are missing the service degrades gracefully to a keyword-similarity fallback so
the API still answers — useful before the first training run.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings
from app.services.catalog import get_catalogue, normalise

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    dish: str
    probability: float


class DishPredictor:
    """Thread-safe inference over the dish classifier."""

    def __init__(self) -> None:
        self.ready = False
        self.reason = "not loaded"
        self.metrics: dict = {}
        self._lock = threading.Lock()
        self._model = None
        self._vectorizer = None
        self._labels: list[str] = []
        self._load()

    # -- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        paths = [settings.model_path, settings.vectorizer_path, settings.labels_path]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            self.reason = (
                f"missing artifacts: {', '.join(missing)}. "
                "Run `python3 -m ml.prepare_data && python3 -m ml.train`."
            )
            log.warning(self.reason)
            return

        try:
            import joblib
            import torch

            from ml.model import DishClassifier, ModelConfig

            self._vectorizer = joblib.load(settings.vectorizer_path)
            self._labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))

            bundle = torch.load(settings.model_path, map_location="cpu", weights_only=False)
            config = ModelConfig.from_dict(bundle["config"])
            model = DishClassifier(config)
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            self._model = model
            self._torch = torch

            if settings.metrics_path.exists():
                self.metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))

            self.ready = True
            self.reason = "ok"
            log.info("classifier loaded: %d labels, %d features", len(self._labels), config.input_dim)
        except Exception as exc:  # pragma: no cover - defensive
            self.reason = f"failed to load model: {exc}"
            log.exception(self.reason)

    def reload(self) -> None:
        with self._lock:
            self.ready = False
            self._load()

    # -- inference ----------------------------------------------------------
    def predict(self, text: str, top_k: int = 5) -> tuple[list[Prediction], float]:
        started = time.perf_counter()
        cleaned = normalise(text)
        if not cleaned:
            return [], 0.0

        if self.ready:
            # PyTorch's grad mode is thread-local and FastAPI runs sync handlers
            # in a worker thread pool, so no_grad() has to be entered per call
            # rather than once at load time.
            with self._lock, self._torch.inference_mode():
                features = self._vectorizer.transform([cleaned])
                dense = self._torch.from_numpy(features.toarray().astype("float32"))
                logits = self._model(dense)[0]
                probs = self._torch.softmax(logits, dim=-1).numpy()
        else:
            probs = self._keyword_fallback(cleaned)

        order = np.argsort(probs)[::-1][:top_k]
        results = [Prediction(self._label(i), float(probs[i])) for i in order]
        return results, (time.perf_counter() - started) * 1000.0

    def distribution(self, text: str) -> dict[str, float]:
        """Full probability vector keyed by dish name, for the Bayesian prior."""
        preds, _ = self.predict(text, top_k=len(self._labels) or 10_000)
        return {p.dish: p.probability for p in preds}

    def _label(self, index: int) -> str:
        if self._labels:
            return self._labels[index]
        return get_catalogue().names[index]

    def _keyword_fallback(self, cleaned: str) -> np.ndarray:
        """Token-overlap scoring, used only when no trained model is present."""
        catalogue = get_catalogue()
        tokens = set(cleaned.split())
        scores = np.zeros(len(catalogue.names), dtype=np.float64)
        for i, name in enumerate(catalogue.names):
            dish = catalogue.get(name)
            bag = set(
                normalise(
                    " ".join(
                        [
                            dish["name"], dish["cuisine"], dish["region"], dish["country"],
                            dish["course"], dish["method"], dish["protein"], dish["carb"],
                            *dish["flags"], *dish["key_ingredients"], *dish["clues"],
                        ]
                    )
                ).split()
            )
            scores[i] = len(tokens & bag) / (len(tokens) + 1e-9)
        if scores.sum() <= 0:
            return np.full(len(scores), 1.0 / len(scores))
        exp = np.exp(scores * 6.0)
        return exp / exp.sum()


@lru_cache(maxsize=1)
def get_predictor() -> DishPredictor:
    return DishPredictor()
