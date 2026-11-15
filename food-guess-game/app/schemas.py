"""Pydantic request/response models — the API contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Answer = Literal["yes", "no", "unsure"]


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class DishCard(BaseModel):
    name: str
    cuisine: str
    region: str
    country: str
    course: str
    difficulty: int
    key_ingredients: list[str]
    fact: str
    emoji: str = "🍽️"


class Candidate(BaseModel):
    dish: str
    probability: float


class SessionRef(BaseModel):
    session_id: str


# --------------------------------------------------------------------------- #
# Mode 1 — the chef hides a dish
# --------------------------------------------------------------------------- #
class NewClueGame(BaseModel):
    client_id: str | None = None
    difficulty_min: int = Field(1, ge=1, le=5)
    difficulty_max: int = Field(5, ge=1, le=5)
    region: str | None = None
    exclude: list[str] = Field(default_factory=list)


class Hint(BaseModel):
    # "clue" is one of the dish's hand-written clues; "trait" is a statement
    # read off the decision tree and removes about half the remaining dishes.
    kind: Literal["clue", "trait"]
    text: str


class ClueGameState(BaseModel):
    session_id: str
    mode: Literal["clue"] = "clue"
    status: str
    clue: str | None = None             # the hint revealed by this call, if any
    hint_kind: Literal["clue", "trait"] | None = None
    clues_revealed: int                 # hand-written clues shown so far
    clues_available: int
    hints_used: int                     # generated traits shown so far
    hints_revealed: int                 # both kinds together
    hints_total: int
    candidates_left: int                # dishes still fitting every trait shown
    wrong_guesses: int
    potential_score: int
    difficulty: int
    region_hint: str | None = None
    message: str | None = None
    history: list[Hint] = Field(default_factory=list)


class GuessIn(BaseModel):
    session_id: str
    guess: str = Field(min_length=1, max_length=200)


class GuessOut(BaseModel):
    correct: bool
    matched: str | None
    match_score: float
    match_method: str
    near_miss: str | None = None
    # Set when the guess contradicts a hint the chef has already given.
    feedback: str | None = None
    state: ClueGameState
    reveal: DishCard | None = None
    score_awarded: int = 0


class RevealOut(BaseModel):
    dish: DishCard
    all_clues: list[str]
    state: ClueGameState


# --------------------------------------------------------------------------- #
# Mode 2 — the chef reads your mind
# --------------------------------------------------------------------------- #
class NewMindGame(BaseModel):
    client_id: str | None = None
    difficulty_min: int = Field(1, ge=1, le=5)
    difficulty_max: int = Field(5, ge=1, le=5)
    region: str | None = None
    # Optional free-text nudge; the classifier turns it into a prior.
    hint: str | None = Field(None, max_length=300)


class MindQuestion(BaseModel):
    key: str
    text: str


class Disagreement(BaseModel):
    """An answer the player gave that differs from the catalogue's entry."""

    key: str
    question: str
    answered: Answer
    catalogue: str


class MindState(BaseModel):
    session_id: str
    mode: Literal["mind"] = "mind"
    status: str
    question: MindQuestion | None = None
    guess: DishCard | None = None
    questions_asked: int
    max_questions: int
    guesses_made: int = 0
    max_guesses: int = 3
    confidence: float
    progress: float
    effective_candidates: float = 0.0   # 2^entropy of the belief
    top_candidates: list[Candidate]
    message: str | None = None
    # Filled once the player has said what the dish really was.
    revealed: DishCard | None = None
    review: list[Disagreement] | None = None


class AnswerIn(BaseModel):
    session_id: str
    key: str
    answer: Answer


class VerdictIn(BaseModel):
    session_id: str
    correct: bool
    # If the engine was wrong the player can tell it what the dish actually was.
    actual: str | None = None


class RevealIn(BaseModel):
    session_id: str
    actual: str = Field(min_length=1, max_length=200)


# --------------------------------------------------------------------------- #
# Mode 3 — describe it, the model predicts
# --------------------------------------------------------------------------- #
class PredictIn(BaseModel):
    text: str = Field(min_length=2, max_length=600)
    top_k: int = Field(5, ge=1, le=20)
    session_id: str | None = None


class PredictOut(BaseModel):
    predictions: list[Candidate]
    latency_ms: float
    model_ready: bool
    prediction_id: int | None = None


class FeedbackIn(BaseModel):
    prediction_id: int | None = None
    session_id: str | None = None
    input_text: str
    predicted_label: str | None = None
    correct_label: str


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
class HealthOut(BaseModel):
    status: str
    version: str
    dishes: int
    questions: int
    model_ready: bool
    model_note: str
    model_metrics: dict | None = None


class AnalyticsOut(BaseModel):
    games_played: int
    games_won: int
    win_rate: float
    avg_clues_to_solve: float | None
    avg_questions_to_read_mind: float | None
    mind_reader_accuracy: float | None
    classifier_feedback_accuracy: float | None
    hardest_dishes: list[dict]
    easiest_dishes: list[dict]
    most_played: list[dict]
