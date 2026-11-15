"""Reverse mode: the chef hides a dish and the player works it out from hints.

The backbone of a round is the hidden dish's own path through a fitted
decision tree (``app.services.tree``). Forward mode asks those questions;
here the chef answers them unprompted, one per hint. Each tree node is the
most even split of the dishes still in play, so each hint removes about half
of them — broad traits first, the distinguishing detail last — and no single
generated hint can give the dish away early.

The dish's four hand-written clues are woven around that ladder:

    clue 1          an oblique opener
    traits ...      the ladder, until only a handful of dishes still fit
    clue 2          the handover
    traits ...      the rest of the ladder, down to one dish
    clue 3, clue 4  the signature details; clue 4 all but names it

A wrong guess changes what comes next. If the guessed dish already
contradicts something the chef has said, the chef points at that hint.
Otherwise the next hint is the split where the guess and the hidden dish part
ways in the tree, which is the one fact that rules the guess out.
"""
from __future__ import annotations

import copy
import random

import numpy as np

from app.config import settings
from app.services.catalog import Catalogue, get_catalogue
from app.services.engine import question_table
from app.services.tree import forest
from ml.validate_catalog import find_leak, leak_terms


def _columns(pool: list[str]) -> np.ndarray:
    index = question_table().name_index
    return np.fromiter((index[n] for n in pool), dtype=np.intp, count=len(pool))


def _fits(key: str, truth: bool, columns: np.ndarray) -> np.ndarray:
    """Which pool dishes agree with the statement ``key is truth``."""
    table = question_table()
    return table.truth[table.key_index[key], columns] == truth


def build_plan(dish: dict, pool: list[str], tree_index: int,
               catalogue: Catalogue | None = None) -> list[dict]:
    """The ordered hints for one round. Pure: same inputs, same plan."""
    catalogue = catalogue or get_catalogue()
    columns = _columns(pool)
    forbidden = leak_terms(dish, include_aliases=True)

    alive = np.ones(len(columns), dtype=bool)
    traits: list[dict] = []
    for split in forest()[tree_index].path(dish["name"]):
        text = catalogue.question_by_key[split.key].fact(split.truth)
        narrowed = alive & _fits(split.key, split.truth, columns)
        if narrowed.sum() == alive.sum():
            continue                      # says nothing new about this pool
        if find_leak(text, forbidden, dish["name"]):
            continue                      # would spell out part of the name
        alive = narrowed
        traits.append({"kind": "trait", "key": split.key, "truth": split.truth,
                       "text": text})

    # Hand over to the written clues once few enough dishes still fit.
    cut, remaining = len(traits), np.ones(len(columns), dtype=bool)
    for position, trait in enumerate(traits):
        remaining &= _fits(trait["key"], trait["truth"], columns)
        if remaining.sum() <= settings.ladder_floor:
            cut = position + 1
            break

    clues = [{"kind": "clue", "index": i, "text": text}
             for i, text in enumerate(dish["clues"])]
    return clues[:1] + traits[:cut] + clues[1:2] + traits[cut:] + clues[2:]


def new_state(dish: dict, pool: list[str], rng: random.Random | None = None) -> dict:
    """Start a round: pick a tree, lay out the plan, reveal the opening clue."""
    rng = rng or random
    tree_index = rng.randrange(len(forest()))
    return {
        "tree": tree_index,
        "plan": build_plan(dish, pool, tree_index),
        "revealed": [0],
        "contrast": None,
    }


def revealed_steps(state: dict) -> list[dict]:
    return [state["plan"][i] for i in state["revealed"]]


def counts(state: dict) -> tuple[int, int]:
    """(written clues revealed, generated traits revealed)."""
    steps = revealed_steps(state)
    clues = sum(1 for s in steps if s["kind"] == "clue")
    return clues, len(steps) - clues


def candidates_left(state: dict, pool: list[str]) -> int:
    """Dishes in the pool that fit every generated trait revealed so far.

    Written clues are prose and are not counted, so this is an upper bound on
    what a careful player could still be considering.
    """
    columns = _columns(pool)
    alive = np.ones(len(columns), dtype=bool)
    for step in revealed_steps(state):
        if step["kind"] == "trait":
            alive &= _fits(step["key"], step["truth"], columns)
    return int(alive.sum())


def _differs(step: dict, other: str) -> bool:
    table = question_table()
    column = table.name_index[other]
    return bool(table.truth[table.key_index[step["key"]], column]) != step["truth"]


def advance(state: dict) -> tuple[dict, dict | None]:
    """Reveal the next hint. Returns the new state and the hint (None when spent).

    After a wrong guess the hint that separates that guess from the hidden dish
    jumps the queue, and is returned with a ``contrast`` naming the guess.
    """
    state = copy.deepcopy(state)
    done = set(state["revealed"])
    pending = [i for i in range(len(state["plan"])) if i not in done]
    other, state["contrast"] = state.get("contrast"), None
    if not pending:
        return state, None

    choice = pending[0]
    contrast = None
    if other:
        for index in pending:
            step = state["plan"][index]
            if step["kind"] == "trait" and _differs(step, other):
                choice, contrast = index, other
                break

    state["revealed"].append(choice)
    hint = dict(state["plan"][choice])
    if contrast:
        hint["contrast"] = contrast
    return state, hint


def judge_wrong_guess(state: dict, guess: str) -> tuple[dict, str | None]:
    """React to a wrong guess that matched a real dish.

    If the guess breaks a hint already given, say which. Otherwise remember the
    guess so the next hint can rule it out.
    """
    state = copy.deepcopy(state)
    if guess not in question_table().name_index:
        return state, None
    for step in revealed_steps(state):
        if step["kind"] == "trait" and _differs(step, guess):
            return state, f"It cannot be {guess}. Remember: {step['text']}"
    state["contrast"] = guess
    return state, None


def hint_text(hint: dict) -> str:
    if hint.get("contrast"):
        return f"Unlike {hint['contrast']}: {hint['text']}"
    return hint["text"]
