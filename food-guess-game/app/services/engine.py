"""Game engines.

``ClueGame``    — the chef hides a dish and doles out clues; the player guesses.
``MindReader``  — the player hides a dish; the engine narrows it down by asking
                  the highest-information question at every step.

The mind reader keeps a full probability distribution over the catalogue rather
than a shrinking candidate list. That matters: a hard-mode player will sometimes
answer wrongly (regional variants, faulty memory), and a hard filter would
permanently eliminate the true dish on the first mistake. A soft Bayesian update
with an explicit noise model lets a dish claw its way back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings
from app.services.catalog import Catalogue, Question, get_catalogue

YES, NO, UNSURE = "yes", "no", "unsure"


# --------------------------------------------------------------------------- #
# Precomputed predicate matrix
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def predicate_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """Boolean matrix of shape (n_questions, n_dishes).

    Evaluating 60-odd Python predicates against 139 dishes on every turn is
    wasteful when the answers never change. Compute it once.
    """
    catalogue = get_catalogue()
    dish_names = catalogue.names
    questions = catalogue.questions
    matrix = np.zeros((len(questions), len(dish_names)), dtype=bool)
    for qi, question in enumerate(questions):
        for di, name in enumerate(dish_names):
            matrix[qi, di] = bool(question.predicate(catalogue.get(name)))
    return matrix, [q.key for q in questions], dish_names


def entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# Belief state
# --------------------------------------------------------------------------- #
@dataclass
class Belief:
    names: list[str]
    probs: np.ndarray

    @classmethod
    def uniform(cls, names: list[str]) -> "Belief":
        return cls(names, np.full(len(names), 1.0 / len(names)))

    @classmethod
    def from_dict(cls, mapping: dict[str, float]) -> "Belief":
        names = list(mapping)
        probs = np.array([mapping[n] for n in names], dtype=np.float64)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(names), 1.0 / len(names))
        return cls(names, probs)

    def to_dict(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.probs)}

    def normalise(self) -> None:
        total = self.probs.sum()
        if total <= 0:
            self.probs = np.full(len(self.probs), 1.0 / len(self.probs))
        else:
            self.probs = self.probs / total

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(self.probs)[::-1][:k]
        return [(self.names[i], float(self.probs[i])) for i in order]

    @property
    def entropy(self) -> float:
        return entropy(self.probs)

    @property
    def confidence(self) -> float:
        return float(self.probs.max()) if len(self.probs) else 0.0


def build_initial_belief(
    pool_names: list[str],
    prior_hint: dict[str, float] | None = None,
    hint_strength: float = 0.6,
) -> Belief:
    """Uniform over the eligible pool, optionally tilted by the classifier.

    ``hint_strength`` blends the model's distribution with the uniform prior.
    Trusting the model completely would let one confident-but-wrong prediction
    dominate before a single question has been asked.
    """
    belief = Belief.uniform(pool_names)
    if prior_hint:
        model = np.array([max(prior_hint.get(n, 0.0), 1e-9) for n in pool_names])
        model = model / model.sum()
        belief.probs = (1 - hint_strength) * belief.probs + hint_strength * model
        belief.normalise()
    return belief


# --------------------------------------------------------------------------- #
# Mind reader
# --------------------------------------------------------------------------- #
class MindReader:
    """Chooses questions by expected information gain and updates by Bayes."""

    def __init__(self, catalogue: Catalogue | None = None) -> None:
        self.catalogue = catalogue or get_catalogue()
        self.matrix, self.question_keys, self.all_names = predicate_matrix()
        self._index_of = {name: i for i, name in enumerate(self.all_names)}

    # -- helpers ------------------------------------------------------------
    def _truth_row(self, question_key: str, names: list[str]) -> np.ndarray:
        qi = self.question_keys.index(question_key)
        cols = [self._index_of[n] for n in names]
        return self.matrix[qi][cols]

    def _likelihood(self, truth: np.ndarray, answer: str) -> np.ndarray:
        """P(answer | dish), given whether the predicate holds for that dish."""
        p_yes = np.where(truth, settings.p_correct, settings.p_flip)
        if answer == YES:
            return p_yes
        if answer == NO:
            return 1.0 - p_yes
        # "unsure" is weakly informative: it very slightly favours dishes whose
        # answer is genuinely debatable, which in practice means it barely moves
        # the belief at all. Modelled as a flat likelihood with a mild penalty
        # applied to the currently dominant candidates, so the game progresses.
        return np.full(truth.shape, settings.unsure_penalty)

    # -- public API ---------------------------------------------------------
    def update(self, belief: Belief, question_key: str, answer: str) -> Belief:
        truth = self._truth_row(question_key, belief.names)
        belief.probs = belief.probs * self._likelihood(truth, answer)
        belief.normalise()
        return belief

    def reject(self, belief: Belief, dish: str) -> Belief:
        """The player said our guess was wrong. Suppress it, don't erase it."""
        if dish in belief.names:
            belief.probs[belief.names.index(dish)] *= 0.02
            belief.normalise()
        return belief

    def information_gain(self, belief: Belief, question_key: str) -> float:
        truth = self._truth_row(question_key, belief.names)
        prior_h = belief.entropy

        p_yes_given = np.where(truth, settings.p_correct, settings.p_flip)
        joint_yes = belief.probs * p_yes_given
        joint_no = belief.probs * (1.0 - p_yes_given)

        mass_yes = joint_yes.sum()
        mass_no = joint_no.sum()
        if mass_yes <= 1e-12 or mass_no <= 1e-12:
            return 0.0

        expected = (
            mass_yes * entropy(joint_yes / mass_yes)
            + mass_no * entropy(joint_no / mass_no)
        )
        return prior_h - expected

    def next_question(
        self, belief: Belief, asked: list[str], jitter: float = 0.02
    ) -> Question | None:
        """Highest expected information gain, with a nudge of randomness.

        Without the jitter every game against the same answers would ask the
        identical sequence, which makes the engine feel scripted.
        """
        asked_set = set(asked)
        best: tuple[float, Question] | None = None
        rng = np.random.default_rng()
        for question in self.catalogue.questions:
            if question.key in asked_set:
                continue
            gain = self.information_gain(belief, question.key)
            if gain <= 1e-6:
                continue
            score = gain + rng.uniform(0, jitter)
            if best is None or score > best[0]:
                best = (score, question)
        return best[1] if best else None

    def should_guess(self, belief: Belief, questions_asked: int) -> bool:
        if belief.confidence >= settings.guess_threshold:
            return True
        # Late in the game, guess as soon as one candidate is clearly ahead.
        if questions_asked >= 12 and belief.confidence >= 0.32:
            return True
        return questions_asked >= settings.max_questions

    def progress(self, belief: Belief) -> float:
        """0 → no idea, 1 → certain. Used for the confidence meter in the UI."""
        max_h = math.log2(len(belief.names)) if len(belief.names) > 1 else 1.0
        return float(max(0.0, min(1.0, 1.0 - belief.entropy / max_h)))


# --------------------------------------------------------------------------- #
# Clue game scoring
# --------------------------------------------------------------------------- #
def clue_score(
    difficulty: int, clues_revealed: int, wrong_guesses: int, hints_used: int
) -> int:
    """Points for solving a dish. Never negative, always rewards a solve."""
    base = settings.base_points + (difficulty - 1) * 25
    penalty = (
        max(0, clues_revealed - 1) * settings.clue_cost
        + wrong_guesses * settings.wrong_guess_cost
        + hints_used * settings.hint_cost
    )
    return max(10, base - penalty)


def streak_bonus(streak: int) -> int:
    return 0 if streak < 2 else min(100, 10 * (streak - 1))
