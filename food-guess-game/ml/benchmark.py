"""Benchmarks for both game modes.

One place for every number the project quotes, shared by the command line
(``python -m tests.simulate``) and the notebook (``notebooks/benchmark.ipynb``).

Forward mode is measured by playing the engine against a scripted player who
knows the truth for every question and answers from the catalogue — except
that a set fraction of answers are flipped or replaced by "not sure". Any
failure to converge is then the engine's fault, not the data's.

What these numbers are not: the scripted player's mistakes are independent
coin flips. A real player's mistakes come from one misconception about the
dish and so are correlated across related questions. Expect live accuracy to
sit below the simulated figure until the noise model is fitted from play logs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.services.catalog import get_catalogue
from app.services.engine import NO, UNSURE, YES, Belief, MindReader, Policy


# --------------------------------------------------------------------------- #
# Scripted players
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Player:
    label: str
    wrong: float = 0.0    # chance a yes/no answer is the opposite of the truth
    unsure: float = 0.0   # chance the player says "not sure" instead of answering
    # Does this player count toward the accuracy target? The target is defined
    # for a player who answers every question and gets up to 10% of them wrong.
    # Everything else is a stress case, reported but not promised.
    in_target: bool = True

    def capacity(self) -> float:
        """Bits a single question can carry past this player's noise."""
        w = self.wrong
        noise = 0.0 if w in (0.0, 1.0) else -(w * math.log2(w) + (1 - w) * math.log2(1 - w))
        return (1.0 - self.unsure) * (1.0 - noise)

    def floor(self, n_dishes: int) -> float:
        """Shannon bound: no engine can average fewer questions than this."""
        return math.log2(n_dishes) / self.capacity()


PLAYERS: list[Player] = [
    Player("honest"),
    Player("5% wrong", wrong=0.05),
    Player("10% wrong", wrong=0.10),
    Player("20% not sure", unsure=0.20, in_target=False),
    Player("10% wrong + 10% not sure", wrong=0.10, unsure=0.10, in_target=False),
    Player("15% wrong", wrong=0.15, in_target=False),
    Player("15% wrong + 15% not sure", wrong=0.15, unsure=0.15, in_target=False),
]

# The engine as it shipped before this redesign, for a like-for-like "before".
LEGACY_POLICY = Policy(
    guess_threshold=0.55, max_questions=20, min_gain_bits=1e-6,
    max_guesses=3, early_rule=(12, 0.32),
)
# Questions added in the redesign that share a kind with the original ones.
ADDED_ATTRIBUTE_QUESTIONS = {
    "protein:fish", "protein:egg", "carb:root", "method:panfried", "temp:room",
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a proportion; behaves near 0 and 1."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# Forward mode
# --------------------------------------------------------------------------- #
@dataclass
class Game:
    target: str
    first_guess_right: bool
    solved: bool
    questions: int
    guesses: int
    first_confidence: float


def play_forward(target: str, reader: MindReader, player: Player, rng: np.random.Generator) -> Game:
    """One round, following exactly the moves the API makes."""
    table = reader.table
    column = table.name_index[target]
    policy = reader.policy

    belief = Belief.uniform(table.names)
    asked: list[str] = []
    rejected: list[str] = []
    questions = guesses = 0
    first_confidence = 0.0

    while True:
        move, what = reader.next_move(belief, asked, questions, rejected)
        if move == "ask":
            asked.append(what.key)
            questions += 1
            if rng.random() < player.unsure:
                answer = UNSURE
            else:
                truth = bool(table.truth[table.key_index[what.key], column])
                if rng.random() < player.wrong:
                    truth = not truth
                answer = YES if truth else NO
            belief = reader.update(belief, what.key, answer)
            continue

        guesses += 1
        if guesses == 1:
            first_confidence = belief.confidence
        if what == target:
            return Game(target, guesses == 1, True, questions, guesses, first_confidence)
        rejected.append(what)
        belief = reader.reject(belief, what)
        if guesses >= policy.max_guesses:
            return Game(target, False, False, questions, guesses, first_confidence)


def _legacy_reader(rng: np.random.Generator) -> MindReader:
    """The pre-redesign engine: original 82 questions, early guessing, cap of 20."""
    reader = MindReader(policy=LEGACY_POLICY, rng=rng)
    keep = np.array([
        (kind == "attribute" and key not in ADDED_ATTRIBUTE_QUESTIONS)
        or key.startswith(("group:", "region:"))
        for key, kind in zip(reader.table.keys, reader.table.kinds)
    ])
    reader._allowed = keep
    return reader


def run_forward(
    player: Player,
    runs: int = 5,
    seed: int = 7,
    policy: Policy | None = None,
    legacy: bool = False,
) -> list[Game]:
    """Play every dish ``runs`` times against one scripted player."""
    rng = np.random.default_rng(seed)
    reader = _legacy_reader(rng) if legacy else MindReader(policy=policy, rng=rng)
    return [
        play_forward(name, reader, player, rng)
        for name in reader.table.names
        for _ in range(runs)
    ]


def summarise(player: Player, games: list[Game]) -> dict:
    n = len(games)
    first = sum(g.first_guess_right for g in games)
    solved = sum(g.solved for g in games)
    questions = np.array([g.questions for g in games])
    low, high = wilson_interval(first, n)
    return {
        "player": player.label,
        "games": n,
        "first_guess": first / n,
        "first_guess_low": low,
        "first_guess_high": high,
        "within_guesses": solved / n,
        "mean_questions": float(questions.mean()),
        "median_questions": float(np.median(questions)),
        "p90_questions": float(np.percentile(questions, 90)),
        "max_questions": int(questions.max()),
        "shannon_floor": player.floor(len(get_catalogue().names)),
        "in_target": player.in_target,
    }


def forward_table(
    players: list[Player] | None = None,
    runs: int = 5,
    seed: int = 7,
    policy: Policy | None = None,
    legacy: bool = False,
) -> list[dict]:
    return [
        summarise(player, run_forward(player, runs, seed, policy, legacy))
        for player in (players or PLAYERS)
    ]


def calibration(games: list[Game], edges: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001)) -> list[dict]:
    """Stated confidence at the first guess against how often that guess was right."""
    rows = []
    for low, high in zip(edges, edges[1:]):
        bucket = [g for g in games if low <= g.first_confidence < high]
        if not bucket:
            continue
        rows.append({
            "confidence": f"{low:.2f}–{min(high, 1.0):.2f}",
            "games": len(bucket),
            "mean_stated": float(np.mean([g.first_confidence for g in bucket])),
            "actually_right": sum(g.first_guess_right for g in bucket) / len(bucket),
        })
    return rows


def stop_reasons(games: list[Game], policy: Policy | None = None) -> list[dict]:
    """Why each round's first guess was made, and how good the guess was.

    A round ends one of three ways: the engine reached its confidence
    threshold, it had nothing left worth asking, or it hit the question cap.
    Only the first is a guess the engine is happy with.
    """
    policy = policy or Policy()
    groups = {
        f"reached {policy.guess_threshold:.2f} confidence": [],
        "ran out of useful questions": [],
        f"hit the {policy.max_questions}-question cap": [],
    }
    confident, exhausted, capped = groups
    for game in games:
        if game.first_confidence >= policy.guess_threshold:
            groups[confident].append(game)
        elif game.questions >= policy.max_questions:
            groups[capped].append(game)
        else:
            groups[exhausted].append(game)
    return [
        {
            "first guess made because the engine": reason,
            "games": len(members),
            "share": len(members) / len(games),
            "first_guess": float(np.mean([g.first_guess_right for g in members])),
            "mean_confidence": float(np.mean([g.first_confidence for g in members])),
            "mean_questions": float(np.mean([g.questions for g in members])),
        }
        for reason, members in groups.items() if members
    ]


def threshold_sweep(
    player: Player,
    thresholds: tuple[float, ...] = (0.55, 0.70, 0.80, 0.90, 0.95, 0.98),
    runs: int = 3,
    seed: int = 7,
) -> list[dict]:
    """The accuracy/questions trade-off as the commit threshold moves."""
    rows = []
    for threshold in thresholds:
        policy = Policy(guess_threshold=threshold)
        row = summarise(player, run_forward(player, runs, seed, policy))
        row["threshold"] = threshold
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# The hard tree, as a baseline
# --------------------------------------------------------------------------- #
def hard_tree_table(players: list[Player] | None = None, runs: int = 5, seed: int = 7) -> list[dict]:
    """Walk the fitted scikit-learn tree with the same scripted players.

    A hard tree takes every answer at face value and has no "not sure" branch,
    so an unsure player is sent down a coin flip. One wrong turn anywhere on
    the path ends at the wrong dish, which is the failure the Bayesian engine
    exists to avoid.
    """
    from app.services.tree import canonical_tree

    tree = canonical_tree()
    table = tree.table
    rows = []
    for player in players or PLAYERS:
        rng = np.random.default_rng(seed)
        right, used = 0, []
        for column, name in enumerate(table.names):
            for _ in range(runs):
                def answer(key: str) -> bool:
                    if rng.random() < player.unsure:
                        return bool(rng.random() < 0.5)
                    truth = bool(table.truth[table.key_index[key], column])
                    return (not truth) if rng.random() < player.wrong else truth

                reached, questions = tree.classify(answer)
                right += reached == name
                used.append(questions)
        n = len(used)
        low, high = wilson_interval(right, n)
        rows.append({
            "player": player.label,
            "games": n,
            "first_guess": right / n,
            "first_guess_low": low,
            "first_guess_high": high,
            "mean_questions": float(np.mean(used)),
            "max_questions": int(max(used)),
            "in_target": player.in_target,
        })
    return rows


# --------------------------------------------------------------------------- #
# Reverse mode
# --------------------------------------------------------------------------- #
def ladder_profile() -> dict:
    """How fast the hint ladder narrows the field, over every dish and every tree.

    Returns one row per generated trait position (how many dishes still fit
    after it) plus the checks that matter: nothing gives the dish away early,
    no hint leaks the name, and the ladder plus written clues always finishes.
    """
    from app.services import hints
    from app.services.tree import forest
    from ml.validate_catalog import find_leak, leak_terms

    catalogue = get_catalogue()
    pool = catalogue.names
    columns = hints._columns(pool)

    left_after: dict[int, list[int]] = {}
    lengths, traits_per_plan, first_unique, unfinished, leaks = [], [], [], 0, 0
    for tree_index in range(len(forest())):
        for name in pool:
            dish = catalogue.get(name)
            plan = hints.build_plan(dish, pool, tree_index)
            forbidden = leak_terms(dish, include_aliases=True)
            alive = np.ones(len(columns), dtype=bool)
            position, unique_at = 0, None
            for step in plan:
                if step["kind"] != "trait":
                    continue
                position += 1
                leaks += bool(find_leak(step["text"], forbidden, dish["name"]))
                alive &= hints._fits(step["key"], step["truth"], columns)
                left_after.setdefault(position, []).append(int(alive.sum()))
                if unique_at is None and alive.sum() == 1:
                    unique_at = position
            lengths.append(len(plan))
            traits_per_plan.append(position)
            unfinished += unique_at is None
            if unique_at is not None:
                first_unique.append(unique_at)

    rows = []
    for position in sorted(left_after):
        left = np.array(left_after[position])
        rows.append({
            "trait": position,
            "rounds_reaching_it": len(left),
            "median_left": float(np.median(left)),
            "worst_left": int(left.max()),
            "already_unique": float((left == 1).mean()),
        })
    return {
        "rows": rows,
        "plans": len(lengths),
        "mean_hints": float(np.mean(lengths)),
        "min_hints": int(min(lengths)),
        "max_hints": int(max(lengths)),
        "mean_traits": float(np.mean(traits_per_plan)),
        "earliest_unique_trait": int(min(first_unique)),
        "ladders_needing_written_clues": unfinished,
        "name_leaks": leaks,
    }


# --------------------------------------------------------------------------- #
# The question bank
# --------------------------------------------------------------------------- #
def bank_summary(kinds: set[str] | None = None, legacy: bool = False) -> dict:
    """Does the bank tell every dish apart, and by how wide a margin."""
    reader = _legacy_reader(np.random.default_rng(0)) if legacy else MindReader(kinds=kinds)
    truth = reader.table.truth[reader._allowed]
    names = reader.table.names

    groups: dict[bytes, list[str]] = {}
    for column, name in enumerate(names):
        groups.setdefault(truth[:, column].tobytes(), []).append(name)
    collisions = [group for group in groups.values() if len(group) > 1]

    answers = truth.T.astype(np.int16)
    distance = (answers[:, None, :] != answers[None, :, :]).sum(axis=2)
    np.fill_diagonal(distance, truth.shape[0] + 1)
    nearest = distance.min(axis=1)
    share = truth.mean(axis=1)
    return {
        "questions": int(truth.shape[0]),
        "balanced_questions": int(((share >= 0.2) & (share <= 0.8)).sum()),
        "collision_groups": len(collisions),
        "dishes_indistinguishable": int(sum(len(g) for g in collisions)),
        "collisions": collisions,
        "nearest_neighbour_median": float(np.median(nearest)),
        "dishes_one_answer_from_another": int((nearest <= 1).sum()),
        "dishes_two_answers_from_another": int((nearest <= 2).sum()),
    }


def describe_engine() -> dict:
    catalogue = get_catalogue()
    kinds: dict[str, int] = {}
    for question in catalogue.questions:
        kinds[question.kind] = kinds.get(question.kind, 0) + 1
    return {
        "dishes": len(catalogue.dishes),
        "questions": len(catalogue.questions),
        "questions_by_kind": kinds,
        "bits_to_identify": math.log2(len(catalogue.dishes)),
        "guess_threshold": settings.guess_threshold,
        "max_questions": settings.max_questions,
        "assumed_wrong_answer_rate": 1.0 - settings.p_correct,
    }
