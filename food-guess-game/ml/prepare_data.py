"""Build the training corpus for the dish classifier.

Two sources feed the corpus:

1. **Catalogue bootstrap** (always available). Every dish in ``data/dishes/*.json``
   is expanded into many synthetic player utterances by sampling its attributes,
   ingredients and clue text through a large template bank. This is what lets the
   model train the moment you clone the repo.

2. **Food.com recipes** (optional). If ``recipes.csv`` from the Kaggle
   "Food.com Recipes and Reviews" dataset is present, real recipe names,
   descriptions, ingredient lists and keyword tags are matched to catalogue
   dishes and appended as genuine, human-written samples. These are weighted
   more heavily than synthetic rows during training.

Usage
-----
    python3 -m ml.prepare_data                     # bootstrap only
    python3 -m ml.prepare_data --recipes path.csv  # + real Food.com data
    python3 -m ml.prepare_data --per-dish 240      # more synthetic samples
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISH_DIR = ROOT / "data" / "dishes"
ARTIFACT_DIR = ROOT / "data" / "artifacts"

# Default locations we will look in for the Kaggle CSV, in order.
RECIPE_CSV_CANDIDATES = [
    ROOT / "data" / "recipes.csv",
    ROOT / "data" / "foodcom" / "recipes.csv",
    Path("/mnt/user-data/uploads/Food Guss Game/data/recipes.csv"),
]

SEED = 20260809


# --------------------------------------------------------------------------- #
# Catalogue loading
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ø", "o").replace("ł", "l").replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_catalogue() -> list[dict]:
    dishes: list[dict] = []
    for path in sorted(DISH_DIR.glob("*.json")):
        dishes.extend(json.loads(path.read_text(encoding="utf-8")))
    if not dishes:
        raise SystemExit(f"no dish files found in {DISH_DIR}")
    return dishes


# --------------------------------------------------------------------------- #
# Synthetic utterance generation
# --------------------------------------------------------------------------- #
REGION_WORDS = {
    "East Asia": ["east asian", "far eastern"],
    "Southeast Asia": ["southeast asian", "south east asian"],
    "South Asia": ["south asian", "subcontinental", "desi"],
    "Middle East": ["middle eastern", "levantine"],
    "North Africa": ["north african", "maghrebi"],
    "West Africa": ["west african"],
    "East Africa": ["east african", "horn of africa"],
    "Southern Africa": ["southern african"],
    "Western Europe": ["western european"],
    "Northern Europe": ["nordic", "scandinavian", "northern european"],
    "Eastern Europe": ["eastern european", "slavic"],
    "Southern Europe": ["mediterranean", "southern european"],
    "British Isles": ["british", "british isles"],
    "North America": ["north american"],
    "Central America": ["central american"],
    "Caribbean": ["caribbean"],
    "South America": ["south american", "latin american"],
    "Oceania": ["pacific", "oceanian"],
}

COURSE_WORDS = {
    "main": ["a main course", "a main dish", "dinner", "a proper meal"],
    "appetizer": ["a starter", "an appetiser", "a small plate"],
    "soup": ["a soup", "a bowl of soup", "something brothy"],
    "dessert": ["a dessert", "a sweet", "pudding"],
    "breakfast": ["breakfast", "a morning dish", "brunch"],
    "snack": ["a snack", "something to eat on the go", "finger food"],
    "side": ["a side dish", "an accompaniment"],
    "salad": ["a salad", "a cold plate"],
}

METHOD_WORDS = {
    "fried": ["fried", "stir fried", "cooked in a hot pan"],
    "deep-fried": ["deep fried", "fried in oil", "crisp fried"],
    "pan-fried": ["pan fried", "cooked on a griddle", "seared in a pan"],
    "grilled": ["grilled", "cooked over charcoal", "barbecued"],
    "steamed": ["steamed", "cooked in a steamer"],
    "boiled": ["boiled", "simmered in liquid", "poached"],
    "stewed": ["stewed", "slow cooked", "simmered for hours"],
    "braised": ["braised", "slow braised", "cooked low and slow"],
    "baked": ["baked", "cooked in the oven", "oven baked"],
    "raw": ["raw", "uncooked", "cured but not cooked"],
    "fermented": ["fermented", "left to ferment", "aged and soured"],
    "roasted": ["roasted", "roasted over heat", "spit roasted"],
    "assembled": ["assembled", "layered together", "put together cold"],
}

FLAG_WORDS = {
    "vegetarian": ["vegetarian", "no meat in it", "meat free"],
    "vegan": ["vegan", "completely plant based", "no dairy or egg"],
    "spicy": ["spicy", "hot", "packs some heat"],
    "sweet": ["sweet", "on the sweet side", "sugary"],
    "dairy": ["has dairy in it", "creamy", "made with milk or butter"],
    "cheese": ["cheesy", "has melted cheese", "loaded with cheese"],
    "soup": ["soupy", "served in broth", "in a bowl of liquid"],
    "noodle": ["noodles in it", "a noodle dish", "has pasta or noodles"],
    "street_food": ["street food", "sold from a cart", "eaten standing up"],
    "raw": ["served raw", "not cooked with heat"],
    "fermented": ["fermented", "funky and sour", "has a fermented ingredient"],
    "handheld": ["eaten with your hands", "handheld", "no cutlery needed"],
    "saucy": ["saucy", "swimming in sauce", "has a rich gravy"],
    "crispy": ["crispy", "crunchy", "shatteringly crisp"],
    "smoky": ["smoky", "cooked over fire", "tastes of charcoal"],
    "festive": ["a festival dish", "eaten at celebrations", "special occasion food"],
    "one_pot": ["a one pot dish", "everything cooked together"],
    "coconut": ["has coconut", "coconut milk in it", "coconut based"],
    "chilli": ["chillies in it", "chilli heavy", "built on chillies"],
    "nuts": ["has nuts", "nutty", "ground nuts in the sauce"],
    "stuffed": ["stuffed", "filled with something", "has a filling inside"],
    "offal": ["uses offal", "made with organ meat", "nose to tail"],
    "pickled": ["something pickled in it", "tangy from pickles", "has a sour pickle"],
    "bread_based": ["bread is part of it", "served on or in bread"],
    "egg_based": ["has egg", "egg is central to it"],
    "seafood": ["seafood", "from the sea", "shellfish or fish"],
    "layered": ["layered", "built up in layers"],
    "sticky": ["sticky", "stringy and sticky"],
    "contains_alcohol": ["cooked with wine or spirits", "has alcohol in the recipe"],
}

CARB_WORDS = {
    "rice": ["served with rice", "rice based", "built on rice"],
    "wheat": ["wheat based", "made with flour", "has a dough"],
    "noodle": ["noodles", "a noodle dish"],
    "potato": ["potatoes in it", "potato based"],
    "corn": ["corn or maize", "made from maize"],
    "root": ["root vegetables", "starchy tubers"],
    "bread": ["served with bread", "bread is part of it"],
    "none": ["no starch with it", "eaten on its own"],
}

PROTEIN_WORDS = {
    "chicken": ["chicken", "poultry"],
    "beef": ["beef", "cow meat"],
    "pork": ["pork", "pig meat"],
    "lamb": ["lamb", "mutton", "goat or lamb"],
    "fish": ["fish"],
    "seafood": ["seafood", "prawns or shellfish"],
    "egg": ["egg"],
    "tofu": ["tofu", "bean curd"],
    "legume": ["beans or lentils", "pulses", "legumes"],
    "mixed": ["several kinds of meat", "mixed meats"],
    "none": ["no protein really", "vegetables only"],
}

OPENERS = [
    "i'm thinking of {body}",
    "it's {body}",
    "something {body}",
    "a dish that is {body}",
    "guess this: {body}",
    "{body}",
    "the one that is {body}",
    "looking for {body}",
    "i want {body}",
    "describe it as {body}",
]


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def synth_utterances(dish: dict, n: int, rng: random.Random) -> list[str]:
    """Generate ``n`` varied natural-language descriptions of one dish."""
    ingredients = [normalise(i) for i in dish["key_ingredients"]]
    flags = list(dish["flags"])
    out: list[str] = []

    # Real clue text is high-signal; include each clue a few times verbatim and
    # a few times in truncated form, so the model sees genuine prose too.
    for clue in dish["clues"]:
        out.append(normalise(clue))
        words = normalise(clue).split()
        if len(words) > 8:
            out.append(" ".join(words[: len(words) // 2]))
            out.append(" ".join(words[len(words) // 2:]))

    # Ingredient-only queries: the single most common way a player describes a dish.
    for _ in range(max(1, n // 4)):
        k = rng.randint(2, min(4, len(ingredients)))
        picked = rng.sample(ingredients, k)
        out.append(rng.choice([
            "it has {}".format(", ".join(picked)),
            "made with {}".format(" and ".join(picked)),
            "{} are the main things in it".format(", ".join(picked)),
            "contains {}".format(", ".join(picked)),
        ]))

    # Attribute mixtures.
    while len(out) < n:
        parts: list[str] = []

        if rng.random() < 0.75 and dish["region"] in REGION_WORDS:
            parts.append(_pick(rng, REGION_WORDS[dish["region"]]))
        if rng.random() < 0.55:
            parts.append(normalise(dish["cuisine"]))
        if rng.random() < 0.45:
            parts.append("from " + normalise(dish["country"]))
        if rng.random() < 0.5:
            parts.append(_pick(rng, COURSE_WORDS[dish["course"]]))
        if rng.random() < 0.5:
            parts.append(_pick(rng, METHOD_WORDS[dish["method"]]))
        if rng.random() < 0.4:
            parts.append(_pick(rng, PROTEIN_WORDS[dish["protein"]]))
        if rng.random() < 0.4:
            parts.append(_pick(rng, CARB_WORDS[dish["carb"]]))
        if dish["temp"] == "cold" and rng.random() < 0.6:
            parts.append("served cold")
        if flags:
            for flag in rng.sample(flags, min(len(flags), rng.randint(1, 3))):
                if flag in FLAG_WORDS:
                    parts.append(_pick(rng, FLAG_WORDS[flag]))
        if ingredients and rng.random() < 0.8:
            k = rng.randint(1, min(3, len(ingredients)))
            parts.append("with " + ", ".join(rng.sample(ingredients, k)))

        if len(parts) < 2:
            continue
        rng.shuffle(parts)
        body = ", ".join(parts)
        out.append(_pick(rng, OPENERS).format(body=body))

    rng.shuffle(out)
    return out[:n]


# --------------------------------------------------------------------------- #
# Food.com integration
# --------------------------------------------------------------------------- #
def _parse_r_vector(value) -> list[str]:
    """Food.com stores list columns as R-style ``c("a", "b")`` strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in {"character(0)", "NA"}:
        return []
    if text.startswith("c(") and text.endswith(")"):
        inner = text[2:-1]
        try:
            parsed = ast.literal_eval("[" + inner + "]")
            return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return re.findall(r'"([^"]*)"', inner)
    if text.startswith('"') and text.endswith('"'):
        return [text.strip('"')]
    return [text]


def build_alias_index(dishes: list[dict]) -> dict[str, str]:
    """Map every searchable surface form to a canonical dish name."""
    index: dict[str, str] = {}
    for dish in dishes:
        for surface in [dish["name"], *dish.get("aliases", [])]:
            key = normalise(surface)
            # Skip aliases so generic they would match thousands of recipes.
            if len(key) < 4 or key in {"mole", "momo", "curry", "meat pie"}:
                continue
            index.setdefault(key, dish["name"])
    return index


def harvest_foodcom(
    csv_path: Path,
    dishes: list[dict],
    max_per_dish: int,
    chunksize: int = 100_000,
) -> tuple[list[dict], dict[str, int]]:
    """Stream recipes.csv and pull out rows whose title matches a catalogue dish."""
    index = build_alias_index(dishes)
    surfaces = sorted(index, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(s) for s in surfaces) + r")\b")

    wanted = ["Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts"]
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    scanned = 0

    reader = pd.read_csv(
        csv_path, usecols=lambda c: c in wanted, chunksize=chunksize, low_memory=False
    )
    for chunk in reader:
        scanned += len(chunk)
        titles = chunk["Name"].astype(str).map(normalise)
        for pos, title in enumerate(titles):
            match = pattern.search(title)
            if not match:
                continue
            label = index[match.group(1)]
            if counts[label] >= max_per_dish:
                continue
            record = chunk.iloc[pos]
            pieces = [
                str(record.get("Name", "")),
                str(record.get("Description", "") or ""),
                " ".join(_parse_r_vector(record.get("RecipeIngredientParts"))),
                " ".join(_parse_r_vector(record.get("Keywords"))),
                str(record.get("RecipeCategory", "") or ""),
            ]
            text = normalise(" ".join(pieces))
            # Strip the answer out of the text so the model cannot cheat by
            # memorising the dish name itself.
            for surface in surfaces:
                if index[surface] == label:
                    text = re.sub(rf"\b{re.escape(surface)}\b", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text.split()) < 6:
                continue
            rows.append({"text": text, "label": label, "source": "foodcom", "weight": 3.0})
            counts[label] += 1
        print(f"  scanned {scanned:,} recipes, kept {len(rows):,}", file=sys.stderr)

    return rows, dict(counts)


def locate_recipes_csv(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for candidate in RECIPE_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", default=None, help="path to Food.com recipes.csv")
    parser.add_argument("--per-dish", type=int, default=180, help="synthetic samples per dish")
    parser.add_argument("--max-foodcom-per-dish", type=int, default=60)
    parser.add_argument("--out", default=str(ARTIFACT_DIR / "corpus.csv.gz"))
    args = parser.parse_args()

    rng = random.Random(SEED)
    dishes = load_catalogue()
    print(f"catalogue: {len(dishes)} dishes")

    rows: list[dict] = []
    for dish in dishes:
        for text in synth_utterances(dish, args.per_dish, rng):
            if text.strip():
                rows.append(
                    {"text": text, "label": dish["name"], "source": "synthetic", "weight": 1.0}
                )
    print(f"synthetic samples: {len(rows):,}")

    csv_path = locate_recipes_csv(args.recipes)
    if csv_path:
        print(f"Food.com dataset found at {csv_path} — harvesting real recipe text")
        real_rows, counts = harvest_foodcom(csv_path, dishes, args.max_foodcom_per_dish)
        rows.extend(real_rows)
        covered = len(counts)
        print(f"Food.com samples: {len(real_rows):,} covering {covered}/{len(dishes)} dishes")
        if covered < len(dishes):
            missing = [d["name"] for d in dishes if d["name"] not in counts]
            print(f"  no Food.com match for {len(missing)} dishes, e.g. {missing[:8]}")
    else:
        print(
            "Food.com recipes.csv not found — training on the catalogue bootstrap only.\n"
            "  Download 'Food.com Recipes and Reviews' from Kaggle and drop recipes.csv\n"
            "  into data/, then re-run this script to fold in real recipe text."
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    frame.to_csv(out_path, index=False, compression="gzip")

    per_label = frame["label"].value_counts()
    meta = {
        "rows": int(len(frame)),
        "labels": int(frame["label"].nunique()),
        "sources": {k: int(v) for k, v in frame["source"].value_counts().items()},
        "min_per_label": int(per_label.min()),
        "max_per_label": int(per_label.max()),
        "foodcom_used": bool(csv_path),
        "foodcom_path": str(csv_path) if csv_path else None,
    }
    (ARTIFACT_DIR / "corpus_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {out_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
