"""The dish catalogue and the question bank derived from it.

The bank is the single attribute hierarchy both game modes read from. Forward
mode (the chef reads your mind) asks its questions; reverse mode (you guess the
chef's dish) states the answers to the same questions as hints. That is why
every question carries two declarative sentences as well as its interrogative.
"""
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

# How a region group reads inside a sentence.
GROUP_PHRASES = {
    "Asia": "Asia",
    "Europe": "Europe",
    "Africa & Middle East": "Africa or the Middle East",
    "Americas & Oceania": "the Americas or Oceania",
}

# Countries that take an article in running text.
COUNTRY_PHRASES = {
    "United States": "the United States",
    "United Kingdom": "the United Kingdom",
    "Philippines": "the Philippines",
    "Netherlands": "the Netherlands",
    "Czech Republic": "the Czech Republic",
}

# A generated question must be true of at least this many dishes, otherwise it
# is a guess at one dish dressed up as a question.
MIN_COUNTRY_SUPPORT = 2
MIN_CUISINE_SUPPORT = 3
MIN_INGREDIENT_SUPPORT = 3

# Spelling variants of one cuisine label.
CUISINE_ALIASES = {"Sichuanese": "Sichuan"}

# Umbrella labels are not asked about. A player thinking of a Punjabi dish
# would rightly answer yes to "is it North Indian?", but the catalogue files it
# under Punjabi only, so the question would be wrong by construction.
UMBRELLA_CUISINES = {
    "Indian", "North Indian", "North Indian street food", "South Indian",
    "Chinese", "Northern Chinese", "Italian",
}
# A cuisine covering this share of its country's dishes just restates the
# country question, so it is skipped.
CUISINE_MAX_COUNTRY_SHARE = 0.5


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ø", "o").replace("ł", "l").replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Question:
    """One yes/no attribute split.

    ``kind`` selects the answer-noise model the engine applies:
    ``attribute`` and ``origin`` are read straight off the dish's structured
    fields; ``cuisine`` labels are fuzzier at the edges (is it Mughlai or
    Hyderabadi?); ``ingredient`` comes from ``key_ingredients``, which lists
    only the four to six defining ingredients, so a "no" there is weaker
    evidence.
    """

    key: str
    text: str
    predicate: Callable[[dict], bool]
    fact_yes: str = ""
    fact_no: str = ""
    kind: str = "attribute"

    def fact(self, truth: bool) -> str:
        return self.fact_yes if truth else self.fact_no


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


# --------------------------------------------------------------------------- #
# Question bank
# --------------------------------------------------------------------------- #
# flag -> (question, statement when true, statement when false)
FLAG_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "vegetarian": ("Is it vegetarian?",
                   "It is vegetarian.", "It is not vegetarian."),
    "vegan": ("Is it fully vegan — no dairy, no egg, no fish sauce?",
              "It is fully vegan.", "It is not vegan."),
    "spicy": ("Is it spicy?", "It is spicy.", "It is not a spicy dish."),
    "sweet": ("Is it sweet?", "It is sweet.", "It is not sweet."),
    "dairy": ("Does dairy play a real part in it?",
              "Dairy plays a real part in it.", "Dairy plays no real part in it."),
    "cheese": ("Is there cheese in it?",
               "There is cheese in it.", "There is no cheese in it."),
    "soup": ("Is it soupy or brothy?",
             "It is soupy or brothy.", "It is not soupy or brothy."),
    "noodle": ("Are noodles or pasta involved?",
               "Noodles or pasta are involved.", "No noodles or pasta are involved."),
    "street_food": ("Would you typically buy it from a street stall?",
                    "You would typically buy it from a street stall.",
                    "It is not typical street-stall food."),
    "raw": ("Is a main component served raw or uncooked?",
            "A main component is served raw.", "Nothing central to it is served raw."),
    "fermented": ("Does it involve fermentation?",
                  "Fermentation is involved.", "No fermentation is involved."),
    "handheld": ("Do you eat it with your hands?",
                 "You eat it with your hands.",
                 "You would not normally eat it with your hands."),
    "saucy": ("Is there a substantial sauce or gravy?",
              "It comes with a substantial sauce or gravy.",
              "There is no substantial sauce or gravy."),
    "crispy": ("Is there something crisp or crunchy about it?",
               "There is something crisp or crunchy about it.",
               "Nothing about it is crisp or crunchy."),
    "smoky": ("Does it get flavour from fire, smoke or charcoal?",
              "It gets flavour from fire, smoke or charcoal.",
              "Fire and smoke play no part in its flavour."),
    "festive": ("Is it associated with festivals or celebrations?",
                "It is associated with festivals or celebrations.",
                "It is everyday food rather than festival food."),
    "one_pot": ("Is it essentially cooked as one pot?",
                "It is essentially cooked in one pot.", "It is not a one-pot dish."),
    "coconut": ("Is coconut involved?",
                "Coconut is involved.", "There is no coconut in it."),
    "chilli": ("Are chillies central to it?",
               "Chillies are central to it.", "Chillies are not central to it."),
    "nuts": ("Are nuts or seeds a significant ingredient?",
             "Nuts or seeds are a significant ingredient.",
             "Nuts and seeds play no significant part."),
    "stuffed": ("Is something stuffed or filled?",
                "Something is stuffed or filled.", "Nothing is stuffed or filled."),
    "offal": ("Does it use offal or organ meat?",
              "It uses offal or organ meat.", "It uses no offal."),
    "pickled": ("Is there something pickled or soured alongside it?",
                "Something pickled or soured comes with it.",
                "Nothing pickled or soured comes with it."),
    "bread_based": ("Is bread part of the dish itself?",
                    "Bread is part of the dish itself.",
                    "Bread is not part of the dish."),
    "egg_based": ("Is egg an important ingredient?",
                  "Egg is an important ingredient.",
                  "Egg is not an important ingredient."),
    "seafood": ("Does it come from the sea?",
                "It comes from the sea.", "It does not come from the sea."),
    "layered": ("Is it built up in distinct layers?",
                "It is built up in distinct layers.", "It is not layered."),
    "sticky": ("Is the texture notably sticky or stringy?",
               "The texture is notably sticky or stringy.",
               "The texture is not sticky or stringy."),
    "contains_alcohol": ("Is wine, beer or spirits used in cooking it?",
                         "Wine, beer or spirits are used in cooking it.",
                         "No alcohol is used in cooking it."),
}

COURSE_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "main": ("Is it a main course?", "It is a main course.", "It is not a main course."),
    "dessert": ("Is it a dessert?", "It is a dessert.", "It is not a dessert."),
    "soup": ("Is it classed as a soup?", "It is classed as a soup.", "It is not a soup."),
    "breakfast": ("Is it eaten at breakfast?",
                  "It is eaten at breakfast.", "It is not a breakfast dish."),
    "snack": ("Is it a snack rather than a full meal?",
              "It is a snack rather than a full meal.", "It is not a snack."),
    "salad": ("Is it a salad?", "It is a salad.", "It is not a salad."),
    "appetizer": ("Is it a starter or small plate?",
                  "It is a starter or small plate.", "It is not a starter."),
    "side": ("Is it a side dish rather than the centrepiece?",
             "It is a side dish rather than the centrepiece.", "It is not a side dish."),
}

# key, question, statement when true, statement when false, predicate
ENUM_QUESTIONS: list[tuple[str, str, str, str, Callable[[dict], bool]]] = [
    ("protein:meatless", "Is it free of any meat or fish?",
     "It contains no meat or fish.", "It contains meat or fish.",
     lambda d: d["protein"] in {"none", "tofu", "legume", "egg"}),
    ("protein:red", "Does it contain red meat — beef, pork or lamb?",
     "It contains red meat.", "It contains no red meat.",
     lambda d: d["protein"] in {"beef", "pork", "lamb"}),
    ("protein:chicken", "Is chicken the main protein?",
     "Chicken is the main protein.", "Chicken is not the main protein.",
     lambda d: d["protein"] == "chicken"),
    ("protein:pork", "Is pork involved?",
     "Pork is involved.", "No pork is involved.",
     lambda d: d["protein"] == "pork"),
    ("protein:beef", "Is beef the main protein?",
     "Beef is the main protein.", "Beef is not the main protein.",
     lambda d: d["protein"] == "beef"),
    ("protein:lamb", "Is lamb or mutton the main protein?",
     "Lamb or mutton is the main protein.", "Lamb is not the main protein.",
     lambda d: d["protein"] == "lamb"),
    ("protein:sea", "Is fish or seafood the main protein?",
     "Fish or seafood is the main protein.", "Fish and seafood are not the main protein.",
     lambda d: d["protein"] in {"fish", "seafood"}),
    ("protein:fish", "Is fish, rather than shellfish, the main protein?",
     "Fish is the main protein.", "Fish is not the main protein.",
     lambda d: d["protein"] == "fish"),
    ("protein:egg", "Is egg the main protein?",
     "Egg is the main protein.", "Egg is not the main protein.",
     lambda d: d["protein"] == "egg"),
    ("protein:pulse", "Are beans, lentils or soy the main protein?",
     "Beans, lentils or soy are the main protein.",
     "Beans, lentils and soy are not the main protein.",
     lambda d: d["protein"] in {"legume", "tofu"}),

    ("carb:rice", "Is rice part of it?",
     "Rice is part of it.", "Rice is not part of it.",
     lambda d: d["carb"] == "rice"),
    ("carb:wheat", "Is there a wheat dough or batter?",
     "There is a wheat dough or batter.", "There is no wheat dough or batter.",
     lambda d: d["carb"] in {"wheat", "bread", "noodle"}),
    ("carb:bread", "Is bread served as part of it?",
     "Bread is served as part of it.", "No bread is served as part of it.",
     lambda d: d["carb"] == "bread"),
    ("carb:potato", "Are potatoes part of it?",
     "Potatoes are part of it.", "Potatoes are not part of it.",
     lambda d: d["carb"] == "potato"),
    ("carb:corn", "Is maize or corn part of it?",
     "Maize or corn is part of it.", "There is no maize or corn in it.",
     lambda d: d["carb"] == "corn"),
    ("carb:root", "Is a root vegetable other than potato central to it?",
     "A root vegetable other than potato is central to it.",
     "No root vegetable other than potato is central to it.",
     lambda d: d["carb"] == "root"),
    ("carb:none", "Is it eaten without any starch or grain?",
     "It is eaten without any starch or grain.", "It includes a starch or grain.",
     lambda d: d["carb"] == "none"),

    ("method:fried", "Is frying the main cooking method?",
     "Frying is the main cooking method.", "Frying is not the main cooking method.",
     lambda d: d["method"] in {"fried", "deep-fried", "pan-fried"}),
    ("method:deepfried", "Is it deep-fried?",
     "It is deep-fried.", "It is not deep-fried.",
     lambda d: d["method"] == "deep-fried"),
    ("method:panfried", "Is it cooked on a griddle or shallow-fried in a pan?",
     "It is cooked on a griddle or shallow-fried in a pan.",
     "It is not griddled or shallow-fried.",
     lambda d: d["method"] == "pan-fried"),
    ("method:steamed", "Is it steamed?", "It is steamed.", "It is not steamed.",
     lambda d: d["method"] == "steamed"),
    ("method:slow", "Is it slow-cooked, stewed or braised?",
     "It is slow-cooked, stewed or braised.", "It is not stewed or braised.",
     lambda d: d["method"] in {"stewed", "braised", "boiled"}),
    ("method:baked", "Is it baked or roasted in an oven?",
     "It is baked or roasted in an oven.", "It is not baked or roasted.",
     lambda d: d["method"] in {"baked", "roasted"}),
    ("method:grilled", "Is it grilled over direct heat?",
     "It is grilled over direct heat.", "It is not grilled.",
     lambda d: d["method"] == "grilled"),
    ("method:nocook", "Is it assembled rather than cooked?",
     "It is assembled rather than cooked.", "It is cooked, not just assembled.",
     lambda d: d["method"] in {"assembled", "raw"}),

    ("temp:hot", "Is it served hot?", "It is served hot.", "It is not served hot.",
     lambda d: d["temp"] == "hot"),
    ("temp:cold", "Is it served cold?", "It is served cold.", "It is not served cold.",
     lambda d: d["temp"] == "cold"),
    ("temp:room", "Is it served at room temperature?",
     "It is served at room temperature.", "It is not served at room temperature.",
     lambda d: d["temp"] == "room"),
]

# Ingredient concepts: slug, noun phrase, plural?, terms, phrases to ignore.
# A concept is true of a dish when any of its ``key_ingredients`` contains one
# of the terms as whole words. Raw ingredient strings are too inconsistent to
# use directly ("green chilli" / "green chili", "egg" / "eggs" / "boiled egg"),
# and the bare pantry — salt, water, sugar, oil, onion — is left out on purpose:
# nobody can say whether onion is "defining", so the answer would be noise.
# Concepts that merely restate a structured field (chicken, rice, coconut,
# chilli, cheese, alcohol) are left out too; asking the same thing twice makes
# the engine count one belief as two pieces of evidence.
INGREDIENT_CONCEPTS: list[tuple[str, str, bool, tuple[str, ...], tuple[str, ...]]] = [
    ("yogurt", "yogurt", False, ("yogurt", "yoghurt", "curd", "dahi"), ()),
    ("butter", "butter", False, ("butter",), ("peanut butter",)),
    ("ghee", "ghee", False, ("ghee",), ()),
    ("cream", "cream", False, ("cream",), ("coconut cream", "ice cream")),
    ("milk", "milk", False, ("milk",), ("coconut milk", "milk solids")),
    ("khoya", "khoya (reduced milk solids)", False,
     ("milk solids", "khoya", "khoa", "mawa"), ()),
    ("paneer", "paneer or fresh cottage cheese", False,
     ("paneer", "cottage cheese", "chhena", "chenna"), ()),
    ("tomato", "tomato", False, ("tomato", "tomatoes"), ()),
    ("garlic", "garlic", False, ("garlic",), ()),
    ("cauliflower", "cauliflower", False, ("cauliflower",), ()),
    ("cabbage", "cabbage", False, ("cabbage",), ()),
    ("spinach", "spinach", False, ("spinach", "palak"), ()),
    ("peas", "green peas", True, ("peas", "pea"), ()),
    ("carrot", "carrot", False, ("carrot", "carrots"), ()),
    ("bell_pepper", "bell pepper", False, ("bell pepper", "bell peppers", "capsicum"), ()),
    ("cucumber", "cucumber", False, ("cucumber",), ()),
    ("mushroom", "mushrooms", True, ("mushroom", "mushrooms", "shiitake"), ()),
    ("eggplant", "aubergine (eggplant)", False, ("eggplant", "aubergine", "brinjal"), ()),
    ("mango", "mango", False, ("mango",), ()),
    ("pineapple", "pineapple", False, ("pineapple",), ()),
    ("citrus", "lemon or lime", False, ("lemon", "lime"), ()),
    ("bean_sprouts", "bean sprouts", True, ("bean sprouts",), ()),
    ("chickpea", "chickpeas or gram flour", True,
     ("chickpea", "chickpeas", "gram flour", "besan", "chana"), ()),
    ("lentil", "lentils (dal)", True, ("dal", "lentil", "lentils"), ()),
    ("beans", "beans", True, ("beans",), ("bean sprouts",)),
    ("saffron", "saffron", False, ("saffron",), ()),
    ("cardamom", "cardamom", False, ("cardamom",), ()),
    ("cumin", "cumin", False, ("cumin", "jeera"), ()),
    ("turmeric", "turmeric", False, ("turmeric",), ()),
    ("curry_leaves", "curry leaves", True, ("curry leaves", "curry leaf"), ()),
    ("mustard", "mustard seed or mustard oil", False, ("mustard",), ()),
    ("tamarind", "tamarind", False, ("tamarind",), ()),
    ("garam_masala", "garam masala", False, ("garam masala",), ()),
    ("sichuan_pepper", "Sichuan peppercorn", False,
     ("sichuan peppercorn", "sichuan peppercorns", "sichuan pepper"), ()),
    ("soy", "soy sauce", False, ("soy sauce", "soya sauce", "soy"), ()),
    ("vinegar", "vinegar", False, ("vinegar",), ()),
    ("sesame", "sesame", False, ("sesame",), ()),
    ("olive_oil", "olive oil", False, ("olive oil",), ()),
    ("peanut", "peanuts", True, ("peanut", "peanuts"), ()),
    ("cashew", "cashews", True, ("cashew", "cashews"), ()),
    ("pistachio_almond", "pistachios or almonds", True,
     ("pistachio", "pistachios", "almond", "almonds"), ()),
    ("semolina", "semolina", False, ("semolina", "sooji", "suji", "rava"), ()),
    ("rice_flour", "rice flour", False, ("rice flour",), ()),
    ("whole_wheat", "whole-wheat flour", False, ("whole wheat", "atta"), ()),
    ("breadcrumbs", "breadcrumbs", True, ("breadcrumbs", "breadcrumb", "panko"), ()),
    ("shrimp", "shrimp or prawns", True, ("shrimp", "prawn", "prawns"), ()),
    ("sausage", "sausage", False, ("sausage", "sausages", "chorizo"), ()),
    ("minced", "minced or ground meat", False,
     ("minced", "mince", "keema", "ground beef", "ground pork", "ground lamb",
      "ground meat"), ()),
]


def cuisine_label(dish: dict) -> str:
    return CUISINE_ALIASES.get(dish["cuisine"], dish["cuisine"])


def ingredient_text(dish: dict) -> str:
    """The dish's key ingredients as one normalised, delimiter-separated string."""
    return " | ".join(normalise(item) for item in dish["key_ingredients"])


def _concept_matcher(terms: tuple[str, ...], ignore: tuple[str, ...]):
    wanted = re.compile(r"\b(?:" + "|".join(re.escape(normalise(t)) for t in terms) + r")\b")
    unwanted = (
        re.compile(r"\b(?:" + "|".join(re.escape(normalise(t)) for t in ignore) + r")\b")
        if ignore else None
    )

    def matches(text: str) -> bool:
        if unwanted is not None:
            text = unwanted.sub(" ", text)
        return wanted.search(text) is not None

    return matches


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:]


def build_question_bank(dishes: list[dict]) -> list[Question]:
    """Assemble yes/no questions, keeping only those that actually split the set.

    Three filters run at construction time rather than on every turn:

    * a question everything answers the same way carries zero information;
    * a generated question true of only one or two dishes is a guess, not a
      question, so country, cuisine and ingredient questions need minimum
      support;
    * a question whose answers duplicate an earlier one is a re-ask in
      different words, and the engine would wrongly treat it as fresh evidence.
    """
    candidates: list[Question] = []

    for flag, (text, yes, no) in FLAG_QUESTIONS.items():
        candidates.append(
            Question(f"flag:{flag}", text, lambda d, f=flag: f in d["flags"], yes, no)
        )

    for course, (text, yes, no) in COURSE_QUESTIONS.items():
        candidates.append(
            Question(f"course:{course}", text, lambda d, c=course: d["course"] == c, yes, no)
        )

    for key, text, yes, no, predicate in ENUM_QUESTIONS:
        candidates.append(Question(key, text, predicate, yes, no))

    for group, members in REGION_GROUPS.items():
        phrase = GROUP_PHRASES.get(group, group)
        candidates.append(
            Question(f"group:{group}", f"Does it come from {phrase}?",
                     lambda d, m=members: d["region"] in m,
                     f"It comes from {phrase}.", f"It does not come from {phrase}.",
                     kind="origin")
        )

    for region in sorted({d["region"] for d in dishes}):
        candidates.append(
            Question(f"region:{region}", f"Is it from {region}?",
                     lambda d, r=region: d["region"] == r,
                     f"It comes from {region}.", f"It does not come from {region}.",
                     kind="origin")
        )

    minimum: dict[str, int] = {}

    for country in sorted({d["country"] for d in dishes}):
        phrase = COUNTRY_PHRASES.get(country, country)
        key = f"country:{country}"
        minimum[key] = MIN_COUNTRY_SUPPORT
        candidates.append(
            Question(key, f"Is it from {phrase}?",
                     lambda d, c=country: d["country"] == c,
                     f"It comes from {phrase}.", f"It does not come from {phrase}.",
                     kind="origin")
        )

    by_cuisine: dict[str, list[dict]] = {}
    for dish in dishes:
        by_cuisine.setdefault(cuisine_label(dish), []).append(dish)
    country_size: dict[str, int] = {}
    for dish in dishes:
        country_size[dish["country"]] = country_size.get(dish["country"], 0) + 1
    for cuisine in sorted(by_cuisine):
        members = by_cuisine[cuisine]
        countries = {d["country"] for d in members}
        if cuisine in UMBRELLA_CUISINES:
            continue
        if len(countries) == 1:
            share = len(members) / country_size[next(iter(countries))]
            if share >= CUISINE_MAX_COUNTRY_SHARE:
                continue
        article = "an" if cuisine[0] in "AEIOU" else "a"
        key = f"cuisine:{cuisine}"
        minimum[key] = MIN_CUISINE_SUPPORT
        candidates.append(
            Question(key, f"Is it {article} {cuisine} speciality?",
                     lambda d, c=cuisine: cuisine_label(d) == c,
                     f"It is {article} {cuisine} speciality.",
                     f"It is not {article} {cuisine} speciality.",
                     kind="cuisine")
        )

    texts = {d["name"]: ingredient_text(d) for d in dishes}
    for slug, noun, plural, terms, ignore in INGREDIENT_CONCEPTS:
        matcher = _concept_matcher(terms, ignore)
        be = "are" if plural else "is"
        key = f"ing:{slug}"
        minimum[key] = MIN_INGREDIENT_SUPPORT
        candidates.append(
            Question(
                key,
                f"{_sentence(be)} {noun} among its defining ingredients?",
                lambda d, m=matcher: m(texts.get(d["name"]) or ingredient_text(d)),
                f"{_sentence(noun)} {be} among its defining ingredients.",
                f"{_sentence(noun)} {be} not among its defining ingredients.",
                kind="ingredient",
            )
        )

    total = len(dishes)
    useful: list[Question] = []
    seen: set[tuple[bool, ...]] = set()
    for question in candidates:
        answers = tuple(bool(question.predicate(d)) for d in dishes)
        yes = sum(answers)
        if not max(1, minimum.get(question.key, 1)) <= yes < total:
            continue
        if answers in seen:
            continue
        seen.add(answers)
        useful.append(question)
    return useful


def dish_emoji(dish: dict) -> str:
    """A plate icon for the reveal card. Cosmetic only — never used as a clue."""
    flags = set(dish["flags"])
    if dish["course"] == "dessert":
        return "🍮"
    if dish["course"] == "soup" or "soup" in flags:
        return "🍲"
    if dish["carb"] == "noodle" or "noodle" in flags:
        return "🍜"
    if dish["course"] == "salad":
        return "🥗"
    if "stuffed" in flags and dish["carb"] == "wheat":
        return "🥟"
    if dish["carb"] == "rice":
        return "🍚"
    if dish["carb"] == "bread" or "bread_based" in flags:
        return "🫓"
    by_protein = {
        "chicken": "🍗", "beef": "🥩", "lamb": "🍖", "pork": "🥓", "fish": "🐟",
        "seafood": "🦐", "egg": "🍳", "legume": "🫘", "tofu": "🥢",
    }
    if dish["protein"] in by_protein:
        return by_protein[dish["protein"]]
    if dish["carb"] == "potato":
        return "🥔"
    return "🍛" if "saucy" in flags else "🍽️"


@lru_cache(maxsize=1)
def get_catalogue() -> Catalogue:
    return Catalogue()
