"""Runtime configuration, read from the environment with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Chef Mystico"
    version: str = "2.0.0"

    root: Path = ROOT
    dish_dir: Path = ROOT / "data" / "dishes"
    artifact_dir: Path = ROOT / "data" / "artifacts"
    frontend_dir: Path = ROOT / "frontend"

    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "FGG_DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'game.db'}"
        )
    )

    # --- game tuning -------------------------------------------------------
    max_questions: int = 20
    guess_threshold: float = 0.55
    # Answer noise model for the Bayesian belief update. Players misremember,
    # and dishes have regional variants, so a single "wrong" answer must never
    # drive a candidate's probability to exactly zero.
    p_correct: float = 0.90
    p_flip: float = 0.10
    unsure_penalty: float = 0.85

    base_points: int = 120
    clue_cost: int = 22
    wrong_guess_cost: int = 12
    hint_cost: int = 15

    cors_origins: tuple[str, ...] = ("*",)

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "model.pt"

    @property
    def vectorizer_path(self) -> Path:
        return self.artifact_dir / "vectorizer.joblib"

    @property
    def labels_path(self) -> Path:
        return self.artifact_dir / "labels.json"

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "metrics.json"


settings = Settings()
