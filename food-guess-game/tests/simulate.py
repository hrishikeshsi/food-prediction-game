"""Play both engines against scripted players and report how they do.

    python3 -m tests.simulate                  # the full benchmark table
    python3 -m tests.simulate --wrong 0.10     # one player: 10% of answers wrong
    python3 -m tests.simulate --unsure 0.15    # one player: "not sure" 15% of the time
    python3 -m tests.simulate --legacy         # the engine as it was before the redesign
    python3 -m tests.simulate --tree           # the hard scikit-learn tree instead
    python3 -m tests.simulate --ladder         # reverse mode: the hint ladder

The scripted player knows the truth for every question and answers from the
catalogue, apart from the stated share of wrong or "not sure" answers, so any
failure to converge is the engine's fault, not the data's. The numbers and
their caveats are discussed in ``notebooks/benchmark.ipynb``.
"""
from __future__ import annotations

import argparse
import sys

from ml.benchmark import (
    PLAYERS,
    Player,
    describe_engine,
    forward_table,
    hard_tree_table,
    ladder_profile,
)


def _print_forward(rows: list[dict]) -> None:
    print(f"  {'player':<26} {'1st guess':>9} {'95% interval':>15} {'in 3':>7} "
          f"{'mean Q':>7} {'p90':>4} {'max':>4} {'floor':>6}")
    for r in rows:
        mark = "" if r["in_target"] else "  (stress case)"
        print(f"  {r['player']:<26} {r['first_guess']:>8.1%} "
              f"[{r['first_guess_low']:>5.1%}, {r['first_guess_high']:>5.1%}] "
              f"{r['within_guesses']:>7.1%} {r['mean_questions']:>7.1f} "
              f"{r['p90_questions']:>4.0f} {r['max_questions']:>4d} {r['shannon_floor']:>6.1f}{mark}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wrong", "--noise", type=float, default=None, dest="wrong")
    parser.add_argument("--unsure", type=float, default=None)
    parser.add_argument("--runs", type=int, default=5, help="repeats per dish")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--legacy", action="store_true", help="replay the previous engine")
    parser.add_argument("--tree", action="store_true", help="benchmark the hard decision tree")
    parser.add_argument("--ladder", action="store_true", help="profile the reverse-mode hints")
    args = parser.parse_args()

    if args.ladder:
        profile = ladder_profile()
        print(f"hint ladder over {profile['plans']} rounds (every dish × every tree)")
        print(f"  hints per round   mean {profile['mean_hints']:.1f}  "
              f"min {profile['min_hints']}  max {profile['max_hints']}")
        print(f"  {'trait':>5} {'median left':>12} {'worst':>6} {'already unique':>15}")
        for row in profile["rows"]:
            print(f"  {row['trait']:>5} {row['median_left']:>12.0f} {row['worst_left']:>6d} "
                  f"{row['already_unique']:>15.1%}")
        print(f"  earliest a ladder pins the dish down: trait {profile['earliest_unique_trait']}")
        print(f"  hints that leak the dish's name: {profile['name_leaks']}")
        return 0

    players = PLAYERS
    if args.wrong is not None or args.unsure is not None:
        wrong, unsure = args.wrong or 0.0, args.unsure or 0.0
        players = [Player(f"{wrong:.0%} wrong, {unsure:.0%} not sure", wrong, unsure)]

    info = describe_engine()
    print(f"{info['dishes']} dishes · {info['questions']} questions · "
          f"{info['bits_to_identify']:.2f} bits to identify one · {args.runs} runs per dish")

    if args.tree:
        print("hard decision tree (scikit-learn), one path, no second guess")
        print(f"  {'player':<26} {'right':>8} {'95% interval':>15} {'mean Q':>7} {'max':>4}")
        for r in hard_tree_table(players, args.runs, args.seed):
            print(f"  {r['player']:<26} {r['first_guess']:>8.1%} "
                  f"[{r['first_guess_low']:>5.1%}, {r['first_guess_high']:>5.1%}] "
                  f"{r['mean_questions']:>7.1f} {r['max_questions']:>4d}")
        return 0

    print("previous engine (82 questions, guess at 0.55, cap 20)" if args.legacy else
          f"Bayesian engine (guess at {info['guess_threshold']}, cap {info['max_questions']})")
    _print_forward(forward_table(players, args.runs, args.seed, legacy=args.legacy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
