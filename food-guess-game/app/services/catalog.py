"""The dish catalogue and the question bank derived from it."""
from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from app.config import settings

REGION_GROUPS = {
    "Asia": {"East Asia", "Southeast Asia", "South Asia"},
    "Europe": {
        "Western Europe", "Northern Europe", "Eastern Europe",
        "Southern Europe", "British Isles",
    },
    "Africa & Middle East": {
        "North Africa", "West Africa", "East Africa", "Southern Africa", "Middle East",
    },
    "Americas & Oceania": {
        "North America", "Central America", "Caribbean", "South America", "Oceania",
    },
}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ø", "o").replace("ł", "l").replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    predicate: Callable[[dict], bool]


class Catalogue:
    """Loads ``data/dishes/*.json`` once and serves lookups over it."""

    def __init__(self, dish_dir=None) -> None:
        directory = dish_dir or settings.dish_dir
        dishes: list[dict] = []
        for path in sorted(directory.glob("*.json")):
            dishes.extend(json.loads(path.read_text(encoding="utf-8")))
        if not dishes:
            raise RuntimeError(f"no dish definitions found under {directory}")

        self.dishes = dishes
        self.by_name = {d["name"]: d for d in dishes}
        self.names = [d["name"] for d in dishes]

        # Every surface form a player might type, mapped to the canonical name.
        self.surface_index: dict[str, str] = {}
        for dish in dishes:
            for surface in [dish["name"], *dish.get("aliases", [])]:
                self.surface_index[normalise(surface)] = dish["name"]

        self.questions = build_question_bank(dishes)
        self.question_by_key = {q.key: q for q in self.questions}

    # -- lookups ------------------------------------------------------------
    def get(self, name: str) -> dict:
        return self.by_name[name]

    def regions(self) -> list[str]:
        return sorted({d["region"] for d in self.dishes})

    def filter(
        self,
        difficulty_min: int = 1,
        difficulty_max: int = 5,
        region: str | None = None,
    ) -> list[dict]:
        pool = [
            d for d in self.dishes if difficulty_min <= d["difficulty"] <= difficulty_max
        ]
        if region:
            allowed = REGION_GROUPS.get(region, {region})
            pool = [d for d in pool if d["region"] in allowed]
        return pool

    def pick(
        self,
        difficulty_min: int = 1,
        difficulty_max: int = 5,
        region: str | None = None,
        exclude: list[str] | None = None,
        rng: random.Random | None = None,
    ) -> dict:
        rng = rng or random
        pool = self.filter(difficulty_min, difficulty_max, region)
        if not pool:
            pool = self.dishes
        fresh = [d for d in pool if d["name"] not in set(exclude or [])]
        return rng.choice(fresh or pool)

    def attribute_hint(self, dish: dict, used: list[str]) -> tuple[str, str] | None:
        """A generated hint, used once the four written clues are exhausted."""
        options: list[tuple[str, str]] = [
            ("region", f"It comes from {dish['region']}."),
            ("course", f"It is served as {dish['course']}."),
            ("method", f"The defining cooking method is: {dish['method']}."),
            ("temp", f"It is eaten {dish['temp']}."),
            ("cuisine", f"The cuisine is described as {dish['cuisine']}."),
            ("country", f"Its home country is {dish['country']}."),
        ]
        for i, ing in enumerate(dish["key_ingredients"][:3]):
            options.append((f"ingredient{i}", f"One key ingredient is {ing}."))
        remaining = [o for o in options if o[0] not in used]
        return remaining[0] if remaining else None


# --------------------------------------------------------------------------- #
# Question bank
# --------------------------------------------------------------------------- #
FLAG_QUESTIONS: dict[str, str] = {
    "vegetarian": "Is it vegetarian?",
    "vegan": "Is it fully vegan — no dairy, no egg, no fish sauce?",
    "spicy": "Is it spicy?",
    "sweet": "Is it sweet?",
    "dairy": "Does dairy play a real part in it?",
    "cheese": "Is there cheese in it?",
    "soup": "Is it soupy or brothy?",
    "noodle": "Are noodles or pasta involved?",
    "street_food": "Would you typically buy it from a street stall?",
    "raw": "Is a main component served raw or uncooked?",
    "fermented": "Does it involve fermentation?",
    "handheld": "Do you eat it with your hands?",
    "saucy": "Is there a substantial sauce or gravy?",
    "crispy": "Is there something crisp or crunchy about it?",
    "smoky": "Does it get flavour from fire, smoke or charcoal?",
    "festive": "Is it associated with festivals or celebrations?",
    "one_pot": "Is it essentially cooked as one pot?",
    "coconut": "Is coconut involved?",
    "chilli": "Are chillies central to it?",
    "nuts": "Are nuts or seeds a significant ingredient?",
    "stuffed": "Is something stuffed or filled?",
    "offal": "Does it use offal or organ meat?",
    "pickled": "Is there something pickled or soured alongside it?",
    "bread_based": "Is bread part of the dish itself?",
    "egg_based": "Is egg an important ingredient?",
    "seafood": "Does it come from the sea?",
    "layered": "Is it built up in distinct layers?",
    "sticky": "Is the texture notably sticky or stringy?",
    "contains_alcohol": "Is wine, beer or spirits used in cooking it?",
}


def build_question_bank(dishes: list[dict]) -> list[Question]:
    """Assemble yes/no questions, keeping only those that actually split the set.

    A question everything answers the same way carries zero information, so it
    is dropped at construction time rather than being scored and rejected on
    every single turn.
    """
    candidates: list[Question] = []

    for flag, text in FLAG_QUESTIONS.items():
        candidates.append(
            Question(f"flag:{flag}", text, lambda d, f=flag: f in d["flags"])
        )

    for course, text in {
        "main": "Is it a main course?",
        "dessert": "Is it a dessert?",
        "soup": "Is it classed as a soup?",
        "breakfast": "Is it eaten at breakfast?",
        "snack": "Is it a snack rather than a full meal?",
        "salad": "Is it a salad?",
        "appetizer": "Is it a starter or small plate?",
        "side": "Is it a side dish rather than the centrepiece?",
    }.items():
        candidates.append(
            Question(f"course:{course}", text, lambda d, c=course: d["course"] == c)
        )

    candidates += [
        Question("protein:meatless", "Is it free of any meat or fish?",
                 lambda d: d["protein"] in {"none", "tofu", "legume", "egg"}),
        Question("protein:red", "Does it contain red meat — beef, pork or lamb?",
                 lambda d: d["protein"] in {"beef", "pork", "lamb"}),
        Question("protein:chicken", "Is chicken the main protein?",
                 lambda d: d["protein"] == "chicken"),
        Question("protein:pork", "Is pork involved?",
                 lambda d: d["protein"] == "pork"),
        Question("protein:beef", "Is beef the main protein?",
                 lambda d: d["protein"] == "beef"),
        Question("protein:lamb", "Is lamb or mutton the main protein?",
                 lambda d: d["protein"] == "lamb"),
        Question("protein:sea", "Is fish or seafood the main protein?",
                 lambda d: d["protein"] in {"fish", "seafood"}),
        Question("protein:pulse", "Are beans, lentils or soy the main protein?",
                 lambda d: d["protein"] in {"legume", "tofu"}),
    ]

    candidates += [
        Question("carb:rice", "Is rice part of it?", lambda d: d["carb"] == "rice"),
        Question("carb:wheat", "Is there a wheat dough or batter?",
                 lambda d: d["carb"] in {"wheat", "bread", "noodle"}),
        Question("carb:bread", "Is bread served as part of it?",
                 lambda d: d["carb"] == "bread"),
        Question("carb:potato", "Are potatoes part of it?", lambda d: d["carb"] == "potato"),
        Question("carb:corn", "Is maize or corn part of it?", lambda d: d["carb"] == "corn"),
        Question("carb:none", "Is it eaten without any starch or grain?",
                 lambda d: d["carb"] == "none"),
    ]

    candidates += [
        Question("method:fried", "Is frying the main cooking method?",
                 lambda d: d["method"] in {"fried", "deep-fried", "pan-fried"}),
        Question("method:deepfried", "Is it deep-fried?",
                 lambda d: d["method"] == "deep-fried"),
        Question("method:steamed", "Is it steamed?", lambda d: d["method"] == "steamed"),
        Question("method:slow", "Is it slow-cooked, stewed or braised?",
                 lambda d: d["method"] in {"stewed", "braised", "boiled"}),
        Question("method:baked", "Is it baked or roasted in an oven?",
                 lambda d: d["method"] in {"baked", "roasted"}),
        Question("method:grilled", "Is it grilled over direct heat?",
                 lambda d: d["method"] == "grilled"),
        Question("method:nocook", "Is it assembled rather than cooked?",
                 lambda d: d["method"] in {"assembled", "raw"}),
    ]

    candidates += [
        Question("temp:hot", "Is it served hot?", lambda d: d["temp"] == "hot"),
        Question("temp:cold", "Is it served cold?", lambda d: d["temp"] == "cold"),
    ]

    for group, members in REGION_GROUPS.items():
        candidates.append(
            Question(f"group:{group}", f"Does it come from {group}?",
                     lambda d, m=members: d["region"] in m)
        )

    for region in sorted({d["region"] for d in dishes}):
        candidates.append(
            Question(f"region:{region}", f"Is it from {region}?",
                     lambda d, r=region: d["region"] == r)
        )

    # Drop degenerate questions: they must split the catalogue at least a little.
    total = len(dishes)
    useful: list[Question] = []
    for question in candidates:
        yes = sum(1 for d in dishes if question.predicate(d))
        if 0 < yes < total:
            useful.append(question)
    return useful


@lru_cache(maxsize=1)
def get_catalogue() -> Catalogue:
    return Catalogue()
