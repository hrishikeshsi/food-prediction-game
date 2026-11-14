"""Play the mind-reading engine against every dish and report how it does.

    python3 -m tests.simulate                 # perfect player
    python3 -m tests.simulate --noise 0.10    # player answers 10% of questions wrongly
    python3 -m tests.simulate --unsure 0.15   # player says "unsure" 15% of the time

This is the honest test of the engine: a scripted player who knows the ground
truth answers every question from the catalogue, so any failure to converge is
the engine's fault, not the data's.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

from app.config import settings
from app.services.catalog import get_catalogue
from app.services.engine import Belief, MindReader


def play(
    target: str,
    reader: MindReader,
    rng: random.Random,
    noise: float,
    unsure_rate: float,
    max_rejections: int = 2,
) -> tuple[bool, int]:
    catalogue = reader.catalogue
    dish = catalogue.get(target)
    belief = Belief.uniform(catalogue.names)
    asked: list[str] = []
    rejected: list[str] = []
    questions = 0

    while questions < settings.max_questions:
        question = reader.next_question(belief, asked)
        if question is None:
            break
        asked.append(question.key)

        truth = question.predicate(dish)
        if rng.random() < unsure_rate:
            answer = "unsure"
        else:
            if rng.random() < noise:
                truth = not truth
            answer = "yes" if truth else "no"

        belief = reader.update(belief, question.key, answer)
        questions += 1

        if reader.should_guess(belief, questions):
            candidate = next(
                (n for n, _ in belief.top(len(belief.names)) if n not in rejected),
                belief.top(1)[0][0],
            )
            if candidate == target:
                return True, questions
            rejected.append(candidate)
            belief = reader.reject(belief, candidate)
            if len(rejected) > max_rejections:
                return False, questions

    return False, questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--unsure", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1, help="repeats per dish")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    catalogue = get_catalogue()
    reader = MindReader(catalogue)
    rng = random.Random(args.seed)

    wins, totals, failures = 0, [], []
    plays = 0
    for dish in catalogue.dishes:
        for _ in range(args.runs):
            ok, asked = play(dish["name"], reader, rng, args.noise, args.unsure)
            plays += 1
            totals.append(asked)
            if ok:
                wins += 1
            else:
                failures.append(dish["name"])

    print(f"player noise={args.noise:.0%}  unsure={args.unsure:.0%}  runs/dish={args.runs}")
    print(f"solved      {wins}/{plays}  ({wins / plays:.1%})")
    print(f"questions   mean {statistics.mean(totals):.1f}  median {statistics.median(totals)}  "
          f"max {max(totals)}")
    if failures:
        unique = sorted(set(failures))
        print(f"missed ({len(unique)}): {', '.join(unique[:15])}"
              + (" ..." if len(unique) > 15 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
