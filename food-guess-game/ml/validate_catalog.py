"""Validate the dish catalogue: schema, controlled vocabularies and clue leakage.

Run:  python3 -m ml.validate_catalog
Exit code 1 if any hard error is found.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISH_DIR = ROOT / "data" / "dishes"

REQUIRED_KEYS = {
    "name", "aliases", "cuisine", "region", "country", "course", "protein",
    "carb", "method", "temp", "difficulty", "flags", "key_ingredients",
    "clues", "fact",
}

COURSES = {"main", "appetizer", "soup", "dessert", "breakfast", "snack", "side", "salad"}
PROTEINS = {"none", "chicken", "beef", "pork", "lamb", "fish", "seafood", "egg", "tofu", "legume", "mixed"}
CARBS = {"none", "rice", "wheat", "noodle", "potato", "corn", "root", "bread"}
METHODS = {
    "fried", "deep-fried", "pan-fried", "grilled", "steamed", "boiled", "stewed",
    "braised", "baked", "raw", "fermented", "roasted", "assembled",
}
TEMPS = {"hot", "cold", "room"}

# Controlled flag vocabulary. Anything outside this set is a hard error, so the
# question engine never has to reason about a typo.
FLAGS = {
    "vegetarian", "vegan", "spicy", "sweet", "dairy", "cheese", "soup", "noodle",
    "street_food", "raw", "fermented", "handheld", "saucy", "crispy", "smoky",
    "festive", "one_pot", "coconut", "chilli", "nuts", "stuffed", "offal",
    "pickled", "bread_based", "egg_based", "seafood", "layered", "sticky",
    "contains_alcohol",
}

# Flags that must be consistent with the enum fields.
IMPLIED = {
    "soup": lambda d: d["course"] == "soup" or d["method"] in {"boiled", "stewed"},
    "seafood": lambda d: d["protein"] in {"fish", "seafood", "mixed"},
    "raw": lambda d: d["method"] in {"raw", "fermented"},
}


def load_all() -> list[dict]:
    dishes: list[dict] = []
    for path in sorted(DISH_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            block = json.load(fh)
        if not isinstance(block, list):
            raise SystemExit(f"{path.name}: top level must be a JSON array")
        for dish in block:
            dish["_source"] = path.name
        dishes.extend(block)
    return dishes


def normalise(text: str) -> str:
    """Fold accents, lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ø", "o").replace("ł", "l").replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Words that are descriptive rather than identifying. A clue may contain these
# even though they appear inside a dish's name or a translated alias.
GENERIC_WORDS = {
    "soup", "stew", "rice", "bread", "cake", "cakes", "pie", "salad", "chicken",
    "beef", "pork", "lamb", "fish", "egg", "eggs", "cheese", "noodles", "noodle",
    "sauce", "the", "and", "with", "style", "meat", "dish", "dumpling",
    "dumplings", "pancake", "sandwich", "curry", "coconut", "potato", "spicy",
    "sweet", "sour", "old", "clothes", "little", "green", "leaf", "leaves",
    "tea", "hot", "brown", "toast", "pepper", "black", "white", "red", "yellow",
    "fried", "grilled", "steamed", "baked", "boiled", "raw", "roast", "roasted",
    "fermented", "pickled", "stuffed", "seed", "seeds", "milk", "cream", "open",
    "faced", "bean", "beans", "melon", "mustard", "ginseng", "herb", "herbs",
    "saffron", "spinach", "lemon", "lime", "octopus", "oxtail", "jute", "pasta",
    "pizza", "burger", "sizzling", "kentucky", "danish", "turkish", "georgian",
    "nepali", "mongolian", "galician", "polish", "german", "japanese", "chinese",
    "korean", "indian", "greek", "swiss", "scottish", "english", "french",
    "mexican", "brazilian", "peruvian", "filipino", "vietnamese", "thai",
}


def leak_terms(dish: dict, include_aliases: bool) -> list[str]:
    """Terms that must not appear in the clue text under inspection."""
    sources = [dish["name"]]
    if include_aliases:
        sources += list(dish.get("aliases", []))

    terms: set[str] = set()
    for raw in sources:
        norm = normalise(raw)
        if not norm:
            continue
        # The whole string is always forbidden.
        terms.add(norm)
        # Individual identifying words are forbidden too, so "Milanese" is
        # caught inside "Risotto alla Milanese".
        for word in norm.split():
            if len(word) > 3 and word not in GENERIC_WORDS:
                terms.add(word)
    return sorted(terms, key=len, reverse=True)


def find_leak(text: str, terms: list[str], canonical: str | None = None) -> str | None:
    """Word-boundary match for general terms; bare substring for the canonical name.

    "Käsespätzle" contains "spätzle" without a word boundary, so a \b-anchored
    search misses it. Compound words are exactly how a clue accidentally gives
    the answer away, so the canonical name is checked as a plain substring.
    """
    haystack = normalise(text)
    if canonical:
        flat = normalise(canonical).replace(" ", "")
        if flat and len(flat) > 4 and flat in haystack.replace(" ", ""):
            return canonical
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", haystack):
            return term
    return None


def main() -> int:
    dishes = load_all()
    errors: list[str] = []
    warnings: list[str] = []

    seen_names: dict[str, str] = {}
    for dish in dishes:
        src = dish.get("_source", "?")
        name = dish.get("name", "<unnamed>")
        tag = f"[{src}] {name}"

        missing = REQUIRED_KEYS - set(dish) - {"_source"}
        extra = set(dish) - REQUIRED_KEYS - {"_source"}
        if missing:
            errors.append(f"{tag}: missing keys {sorted(missing)}")
        if extra:
            errors.append(f"{tag}: unexpected keys {sorted(extra)}")
        if missing:
            continue

        key = normalise(name)
        if key in seen_names:
            errors.append(f"{tag}: duplicate of a dish already in {seen_names[key]}")
        seen_names[key] = src

        if dish["course"] not in COURSES:
            errors.append(f"{tag}: bad course {dish['course']!r}")
        if dish["protein"] not in PROTEINS:
            errors.append(f"{tag}: bad protein {dish['protein']!r}")
        if dish["carb"] not in CARBS:
            errors.append(f"{tag}: bad carb {dish['carb']!r}")
        if dish["method"] not in METHODS:
            errors.append(f"{tag}: bad method {dish['method']!r}")
        if dish["temp"] not in TEMPS:
            errors.append(f"{tag}: bad temp {dish['temp']!r}")
        if not isinstance(dish["difficulty"], int) or not 1 <= dish["difficulty"] <= 5:
            errors.append(f"{tag}: difficulty must be int 1-5, got {dish['difficulty']!r}")

        bad_flags = set(dish["flags"]) - FLAGS
        if bad_flags:
            errors.append(f"{tag}: unknown flags {sorted(bad_flags)}")
        if len(set(dish["flags"])) != len(dish["flags"]):
            warnings.append(f"{tag}: duplicate flags")

        for flag, predicate in IMPLIED.items():
            if flag in dish["flags"] and not predicate(dish):
                warnings.append(f"{tag}: flag {flag!r} is inconsistent with the enum fields")

        if not 3 <= len(dish["key_ingredients"]) <= 7:
            warnings.append(f"{tag}: {len(dish['key_ingredients'])} key ingredients (expected 4-6)")
        if len(dish["clues"]) != 4:
            errors.append(f"{tag}: expected 4 clues, got {len(dish['clues'])}")

        # Clues 1-3 must not leak the name or any alias.
        # Clue 4 is the deliberate near-giveaway: translated/descriptive aliases
        # are allowed there, but the canonical name still is not.
        # The fact is only ever shown after the reveal, so it is not checked.
        if len(dish["clues"]) == 4:
            early = leak_terms(dish, include_aliases=True)
            for idx, clue in enumerate(dish["clues"][:3]):
                hit = find_leak(clue, early, dish["name"])
                if hit:
                    errors.append(f"{tag}: clue {idx + 1} leaks the answer term {hit!r}")
            hit = find_leak(dish["clues"][3], leak_terms(dish, include_aliases=False), dish["name"])
            if hit:
                errors.append(f"{tag}: clue 4 leaks the canonical name term {hit!r}")

    print(f"dishes loaded: {len(dishes)}")
    print(f"regions:  {dict(Counter(d.get('region') for d in dishes))}")
    print(f"difficulty: {dict(sorted(Counter(d.get('difficulty') for d in dishes).items()))}")
    flag_counts = Counter(f for d in dishes for f in d.get("flags", []))
    print(f"flag usage: {dict(flag_counts.most_common())}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
