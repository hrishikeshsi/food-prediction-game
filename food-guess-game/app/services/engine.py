"""Forward-mode engine: the player hides a dish, the engine finds it.

``MindReader`` is a decision tree grown one node at a time. At every node it
scores each unasked question by expected information gain and asks the best
one — the same split criterion ID3 uses — but the tree it walks has *soft*
splits. A hard tree deletes every dish that disagrees with an answer, so a
single mistaken answer loses the true dish for good: at 10 % wrong answers
only about a third of ten-question rounds survive. Here each answer instead
reweights a probability distribution over the whole catalogue by Bayes' rule,
under an explicit answer-noise model, so a wrongly demoted dish can climb back.

The fitted scikit-learn tree in ``app.services.tree`` is the hard-split
counterpart over the same question table; reverse mode walks that one.

The stopping rule is what makes the accuracy target hold. The engine commits
to a guess only when the leading dish's posterior reaches
``settings.guess_threshold``. If the noise model matches the player, the
posterior is calibrated, so the first guess is right at least that often by
construction. The price is paid in questions, never in accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import settings
from app.services.catalog import Catalogue, Question, get_catalogue

YES, NO, UNSURE = "yes", "no", "unsure"


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in bits of a yes/no outcome with P(yes) = p, elementwise."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# Precomputed question table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuestionTable:
    """Everything the engine needs about the bank, computed once.

    ``truth[q, d]``         does question q hold for dish d, per the catalogue
    ``p_yes[q, d]``         P(player answers yes | they are thinking of d)
    ``noise_entropy[q, d]`` entropy of that answer, the part no question can remove
    """

    truth: np.ndarray
    p_yes: np.ndarray
    noise_entropy: np.ndarray
    keys: list[str]
    kinds: list[str]
    names: list[str]
    key_index: dict[str, int]
    name_index: dict[str, int]


def _reliability(kind: str) -> tuple[float, float]:
    """(P(yes | catalogue says yes), P(yes | catalogue says no)) for a question kind."""
    if kind == "ingredient":
        return settings.ingredient_p_correct, settings.ingredient_p_flip
    if kind == "cuisine":
        return settings.cuisine_p_correct, settings.cuisine_p_flip
    return settings.p_correct, settings.p_flip


@lru_cache(maxsize=1)
def question_table() -> QuestionTable:
    catalogue = get_catalogue()
    names = catalogue.names
    questions = catalogue.questions

    truth = np.zeros((len(questions), len(names)), dtype=bool)
    for qi, question in enumerate(questions):
        for di, name in enumerate(names):
            truth[qi, di] = bool(question.predicate(catalogue.get(name)))

    p_true = np.array([_reliability(q.kind)[0] for q in questions])[:, None]
    p_false = np.array([_reliability(q.kind)[1] for q in questions])[:, None]
    p_yes = np.where(truth, p_true, p_false)

    keys = [q.key for q in questions]
    return QuestionTable(
        truth=truth,
        p_yes=p_yes,
        noise_entropy=binary_entropy(p_yes),
        keys=keys,
        kinds=[q.kind for q in questions],
        names=names,
        key_index={k: i for i, k in enumerate(keys)},
        name_index={n: i for i, n in enumerate(names)},
    )


def predicate_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """Boolean truth table of shape (n_questions, n_dishes), with its two axes."""
    table = question_table()
    return table.truth, table.keys, table.names


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
@dataclass(frozen=True)
class Policy:
    """When to stop asking and commit. Defaults come from ``app.config``."""

    guess_threshold: float = settings.guess_threshold
    max_questions: int = settings.max_questions
    min_gain_bits: float = settings.min_gain_bits
    max_guesses: int = settings.max_guesses
    # (questions asked, confidence): guess early once both are reached. Unused
    # by the live game; kept so the benchmark can replay the previous engine.
    early_rule: tuple[int, float] | None = None


class MindReader:
    """Chooses questions by expected information gain and updates by Bayes."""

    def __init__(
        self,
        catalogue: Catalogue | None = None,
        policy: Policy | None = None,
        kinds: set[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.catalogue = catalogue or get_catalogue()
        self.table = question_table()
        self.policy = policy or Policy()
        self.rng = rng or np.random.default_rng()
        # Optionally restrict the bank to some question kinds (benchmarking).
        self._allowed = np.array(
            [kinds is None or kind in kinds for kind in self.table.kinds], dtype=bool
        )

    # -- helpers ------------------------------------------------------------
    def _columns(self, names: list[str]) -> np.ndarray | None:
        """Catalogue columns for a belief's pool; None when it is the whole catalogue."""
        if names is self.table.names or names == self.table.names:
            return None
        return np.fromiter(
            (self.table.name_index[n] for n in names), dtype=np.intp, count=len(names)
        )

    def _rows(self, array: np.ndarray, names: list[str]) -> np.ndarray:
        columns = self._columns(names)
        return array if columns is None else array[:, columns]

    # -- public API ---------------------------------------------------------
    def update(self, belief: Belief, question_key: str, answer: str) -> Belief:
        """Bayes' rule: multiply by P(answer | dish) and renormalise.

        "Not sure" is equally likely whatever the dish, so it changes nothing —
        but the question is still spent, which is the cost of not knowing.
        """
        if answer == UNSURE:
            return belief
        qi = self.table.key_index[question_key]
        columns = self._columns(belief.names)
        p_yes = self.table.p_yes[qi] if columns is None else self.table.p_yes[qi, columns]
        belief.probs = belief.probs * (p_yes if answer == YES else 1.0 - p_yes)
        belief.normalise()
        return belief

    def reject(self, belief: Belief, dish: str) -> Belief:
        """The player said outright that our guess was wrong.

        A rejected dish is never offered again, so its probability is handed to
        the remaining candidates rather than left to dilute them.
        """
        if dish in belief.names:
            belief.probs[belief.names.index(dish)] = 0.0
            belief.normalise()
        return belief

    def gains(self, belief: Belief, asked: list[str] | set[str] = ()) -> np.ndarray:
        """Expected information gain of every question, in bits.

        The mutual information between the player's answer A and the dish D:

            I(A; D) = H(A) − H(A | D)
                    = h( Σ_d b_d · p_yes[q, d] ) − Σ_d b_d · h( p_yes[q, d] )

        with h the binary entropy. The first term rewards an even split of the
        belief; the second subtracts the part of the answer that is only noise.
        Already-asked and disallowed questions score zero.
        """
        b = belief.probs
        p_yes = self._rows(self.table.p_yes, belief.names)
        noise = self._rows(self.table.noise_entropy, belief.names)
        gains = binary_entropy(p_yes @ b) - noise @ b
        gains = np.maximum(gains, 0.0)
        gains[~self._allowed] = 0.0
        for key in asked:
            index = self.table.key_index.get(key)
            if index is not None:
                gains[index] = 0.0
        return gains

    def information_gain(self, belief: Belief, question_key: str) -> float:
        return float(self.gains(belief)[self.table.key_index[question_key]])

    def next_question(
        self, belief: Belief, asked: list[str], jitter: float = 0.03
    ) -> Question | None:
        """The question with the highest expected information gain.

        ``jitter`` scales each score by up to that fraction, so near-ties are
        broken at random and repeat rounds do not ask an identical sequence.
        Returns None once nothing left is worth asking.
        """
        gains = self.gains(belief, asked)
        if gains.max() < self.policy.min_gain_bits:
            return None
        scores = gains * (1.0 + self.rng.uniform(0.0, jitter, gains.shape))
        return self.catalogue.questions[int(scores.argmax())]

    def should_guess(self, belief: Belief, questions_asked: int) -> bool:
        confidence = belief.confidence
        if confidence >= self.policy.guess_threshold:
            return True
        early = self.policy.early_rule
        if early and questions_asked >= early[0] and confidence >= early[1]:
            return True
        return questions_asked >= self.policy.max_questions

    def best_guess(self, belief: Belief, rejected: list[str] | set[str] = ()) -> str:
        """The most probable dish that has not already been turned down."""
        rejected = set(rejected)
        for index in np.argsort(belief.probs)[::-1]:
            if belief.names[index] not in rejected:
                return belief.names[index]
        return belief.names[int(belief.probs.argmax())]

    def next_move(
        self,
        belief: Belief,
        asked: list[str],
        questions_asked: int,
        rejected: list[str] | set[str] = (),
    ) -> tuple[str, Question | str]:
        """What the engine does next: ``("ask", question)`` or ``("guess", dish)``.

        It guesses when it is confident enough, when it has hit the question
        cap, or when nothing left is worth asking; otherwise it asks. The API
        and the benchmark both go through here, so they cannot drift apart.
        """
        if not self.should_guess(belief, questions_asked):
            question = self.next_question(belief, asked)
            if question is not None:
                return "ask", question
        return "guess", self.best_guess(belief, rejected)

    def progress(self, belief: Belief) -> float:
        """0 → no idea, 1 → certain. Used for the confidence meter in the UI."""
        max_h = math.log2(len(belief.names)) if len(belief.names) > 1 else 1.0
        return float(max(0.0, min(1.0, 1.0 - belief.entropy / max_h)))

    def effective_candidates(self, belief: Belief) -> float:
        """2^entropy: how many equally likely dishes the belief is worth."""
        return float(2.0 ** belief.entropy)

    def disagreements(self, dish: str, answers: list[tuple[str, str]]) -> list[dict]:
        """Where the player's yes/no answers differ from the catalogue for ``dish``."""
        if dish not in self.table.name_index:
            return []
        column = self.table.name_index[dish]
        found = []
        for key, answer in answers:
            index = self.table.key_index.get(key)
            if index is None or answer == UNSURE:
                continue
            truth = bool(self.table.truth[index, column])
            if (answer == YES) != truth:
                question = self.catalogue.questions[index]
                found.append({
                    "key": key,
                    "question": question.text,
                    "answered": answer,
                    "catalogue": question.fact(truth),
                })
        return found


# --------------------------------------------------------------------------- #
# Clue game scoring
# --------------------------------------------------------------------------- #
def clue_score(
    difficulty: int, clues_revealed: int, wrong_guesses: int, hints_used: int
) -> int:
    """Points for solving a dish. Never negative, always rewards a solve.

    A hand-written clue costs more than a generated trait because it says
    more: a trait is worth about one bit, a late clue nearly names the dish.
    """
    base = settings.base_points + (difficulty - 1) * 25
    penalty = (
        max(0, clues_revealed - 1) * settings.clue_cost
        + wrong_guesses * settings.wrong_guess_cost
        + hints_used * settings.hint_cost
    )
    return max(10, base - penalty)


def streak_bonus(streak: int) -> int:
    return 0 if streak < 2 else min(100, 10 * (streak - 1))
