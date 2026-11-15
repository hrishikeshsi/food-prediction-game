"""Runtime configuration, read from the environment with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Chef Mystico"
    version: str = "3.0.0"

    root: Path = ROOT
    dish_dir: Path = ROOT / "data" / "dishes"
    artifact_dir: Path = ROOT / "data" / "artifacts"
    frontend_dir: Path = ROOT / "frontend"

    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "FGG_DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'game.db'}"
        )
    )

    # --- forward mode: the chef reads your mind ------------------------------
    # Hard ceiling on questions per round. At 10 % wrong answers a binary
    # question carries about 0.53 bits, so naming one of 350 dishes needs ~16
    # questions on average; 40 leaves room for the unlucky rounds.
    max_questions: int = 40
    # Commit to a guess once the leading dish holds this much posterior
    # probability. With a calibrated noise model the first guess is then right
    # at least this often — the threshold *is* the accuracy target.
    guess_threshold: float = 0.95
    # ...or once no remaining question is expected to teach us this many bits.
    min_gain_bits: float = 0.02
    # Guesses the engine may make before it concedes the round.
    max_guesses: int = 3

    # Answer-noise model for the Bayesian belief update. Players misremember,
    # and dishes have regional variants, so a single "wrong" answer must never
    # drive a candidate's probability to exactly zero.
    #   p_correct  P(player says yes | the catalogue says yes)
    #   p_flip     P(player says yes | the catalogue says no)
    p_correct: float = 0.90
    p_flip: float = 0.10
    # Ingredient questions are read off ``key_ingredients``, which lists only a
    # dish's four to six defining ingredients. "Not listed" is therefore weaker
    # evidence than "absent", and the model says so. These two are assumptions,
    # not fitted values; ``GET /api/analytics/answer-noise`` reports what real
    # players actually do so they can be replaced.
    ingredient_p_correct: float = 0.85
    ingredient_p_flip: float = 0.20
    # Regional-cuisine labels blur at the edges, in both directions.
    cuisine_p_correct: float = 0.85
    cuisine_p_flip: float = 0.15

    # --- reverse mode: you guess the chef's dish ----------------------------
    # Randomised decision trees the hint ladder draws from, so that two rounds
    # do not open with the same trait. Tree 0 is the canonical exported tree.
    hint_trees: int = 6
    hint_tree_max_features: float = 0.4
    # The generated ladder hands over to the hand-written clues once this few
    # dishes still fit everything said so far.
    ladder_floor: int = 4

    base_points: int = 120
    clue_cost: int = 20          # each hand-written clue after the first
    hint_cost: int = 6           # each generated trait (about one bit)
    wrong_guess_cost: int = 10

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
