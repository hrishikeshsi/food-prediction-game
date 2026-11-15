"""End-to-end tests. Run with:  python3 -m pytest tests -q"""
from __future__ import annotations

import os
import random
import tempfile

import numpy as np
import pytest

# Point the app at a throwaway database before anything imports the engine.
_TMP = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ.setdefault("FGG_DATABASE_URL", f"sqlite:///{_TMP.name}")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.db_models import GameSession  # noqa: E402
from app.main import app  # noqa: E402
from app.services import hints  # noqa: E402
from app.services.catalog import get_catalogue, normalise  # noqa: E402
from app.services.engine import Belief, MindReader, question_table  # noqa: E402
from app.services.matching import match_guess  # noqa: E402
from app.services.tree import canonical_tree, forest  # noqa: E402
from ml.benchmark import PLAYERS, bank_summary, forward_table  # noqa: E402
from ml.validate_catalog import find_leak, leak_terms  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _secret(session_id: str) -> str:
    """The hidden dish. Not available through the API by design, so read the row."""
    with SessionLocal() as db:
        return db.get(GameSession, session_id).secret_dish


# --------------------------------------------------------------------------- #
# catalogue and question bank
# --------------------------------------------------------------------------- #
def test_catalogue_loads():
    cat = get_catalogue()
    assert len(cat.dishes) >= 100
    assert len(cat.questions) >= 40
    assert len(cat.by_name) == len(cat.dishes), "dish names must be unique"


def test_every_dish_is_reachable_by_some_question():
    """No dish may be indistinguishable from another under the whole bank."""
    cat = get_catalogue()
    signatures: dict[tuple, list[str]] = {}
    for dish in cat.dishes:
        sig = tuple(q.predicate(dish) for q in cat.questions)
        signatures.setdefault(sig, []).append(dish["name"])
    collisions = {k: v for k, v in signatures.items() if len(v) > 1}
    assert not collisions, f"dishes with identical answers: {list(collisions.values())}"


def test_question_bank_is_well_formed():
    cat = get_catalogue()
    kinds = {q.kind for q in cat.questions}
    assert {"attribute", "origin", "cuisine", "ingredient"} <= kinds
    assert len({q.key for q in cat.questions}) == len(cat.questions)

    patterns = set()
    for question in cat.questions:
        answers = tuple(question.predicate(d) for d in cat.dishes)
        assert 0 < sum(answers) < len(answers), f"{question.key} does not split the catalogue"
        assert answers not in patterns, f"{question.key} repeats another question's answers"
        patterns.add(answers)
        # Reverse mode states every question as a fact, in both directions.
        assert question.fact_yes and question.fact_no and question.fact_yes != question.fact_no
        if question.kind == "ingredient":
            assert sum(answers) >= 3


def test_ingredient_concepts_match_whole_words_only():
    cat = get_catalogue()
    milk = cat.question_by_key["ing:milk"].predicate
    base = dict(cat.dishes[0])
    assert milk({**base, "name": "t1", "key_ingredients": ["whole milk", "sugar"]})
    assert not milk({**base, "name": "t2", "key_ingredients": ["coconut milk", "rice"]})
    assert not milk({**base, "name": "t3", "key_ingredients": ["buttermilk", "rice"]})
    peas = cat.question_by_key["ing:peas"].predicate
    assert not peas({**base, "name": "t4", "key_ingredients": ["chickpeas"]})


def test_clues_never_contain_the_answer():
    cat = get_catalogue()
    for dish in cat.dishes:
        name = normalise(dish["name"])
        for clue in dish["clues"]:
            assert name not in normalise(clue), f"{dish['name']} leaks in its own clue"


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "typed,expected",
    [
        ("Bigos", "Bigos"),
        ("bigos", "Bigos"),
        ("  BIGOS  ", "Bigos"),
        ("hunter's stew", "Bigos"),
        ("pierogi", "Pierogi"),
        ("mapo tofu", "Mapo Tofu"),
        ("mapo dofu", "Mapo Tofu"),          # misspelling
        ("i think it's poutine", "Poutine"),  # sentence
        ("Rösti", "Rösti"),
        ("rosti", "Rösti"),                   # accent dropped
    ],
)
def test_guess_matching(typed, expected):
    result = match_guess(typed, get_catalogue())
    assert result.accepted, f"{typed!r} was not accepted (score {result.score})"
    assert result.dish == expected


def test_nonsense_guess_is_rejected():
    result = match_guess("qwertyuiop asdfgh", get_catalogue())
    assert not result.accepted


# --------------------------------------------------------------------------- #
# forward mode: the Bayesian engine
# --------------------------------------------------------------------------- #
def _play_honestly(reader: MindReader, target: str) -> tuple[str, int]:
    dish = reader.catalogue.get(target)
    belief = Belief.uniform(reader.catalogue.names)
    asked: list[str] = []
    while True:
        move, what = reader.next_move(belief, asked, len(asked))
        if move == "guess":
            return what, len(asked)
        asked.append(what.key)
        belief = reader.update(belief, what.key, "yes" if what.predicate(dish) else "no")


def test_mind_reader_solves_an_honest_player():
    reader = MindReader(rng=np.random.default_rng(1))
    for target in ["Bigos", "Mohinga", "Poutine", "Lahpet Thoke", "Tahchin", "Garlic Naan"]:
        guess, questions = _play_honestly(reader, target)
        assert guess == target, f"failed to converge on {target}"
        assert questions <= 20


def test_engine_only_guesses_when_confident_or_out_of_road():
    reader = MindReader(rng=np.random.default_rng(2))
    belief = Belief.uniform(reader.catalogue.names)
    move, _ = reader.next_move(belief, [], 0)
    assert move == "ask", "a uniform belief is nowhere near the guess threshold"
    assert reader.next_move(belief, [], settings.max_questions)[0] == "guess"

    sure = Belief.uniform(reader.catalogue.names)
    sure.probs[:] = 0.01 / (len(sure.probs) - 1)
    sure.probs[3] = 0.99
    assert reader.next_move(sure, [], 5) == ("guess", reader.catalogue.names[3])


def test_information_gain_matches_the_definition():
    """The vectorised gain must equal H(belief) − E[H(belief | answer)]."""
    reader = MindReader()
    table = reader.table
    rng = np.random.default_rng(0)
    belief = Belief(table.names, rng.dirichlet(np.ones(len(table.names))))
    gains = reader.gains(belief)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    for index in (0, 17, 60, len(table.keys) - 1):
        yes = belief.probs * table.p_yes[index]
        no = belief.probs * (1 - table.p_yes[index])
        expected = entropy(belief.probs) - (
            yes.sum() * entropy(yes / yes.sum()) + no.sum() * entropy(no / no.sum())
        )
        assert gains[index] == pytest.approx(expected, abs=1e-9)


def test_belief_stays_normalised_and_never_zeroes_the_truth():
    cat = get_catalogue()
    reader = MindReader(cat)
    belief = Belief.uniform(cat.names)
    dish = cat.get("Bigos")
    # Answer the first six questions the wrong way round on purpose.
    asked: list[str] = []
    for _ in range(6):
        q = reader.next_question(belief, asked)
        asked.append(q.key)
        belief = reader.update(belief, q.key, "no" if q.predicate(dish) else "yes")
    assert abs(belief.probs.sum() - 1.0) < 1e-9
    assert belief.to_dict()["Bigos"] > 0, "a lying player must not erase the truth"


def test_unsure_answers_leave_the_belief_alone():
    reader = MindReader()
    belief = Belief.uniform(reader.catalogue.names)
    question = reader.next_question(belief, [])
    belief = reader.update(belief, question.key, "yes")
    before = belief.probs.copy()
    belief = reader.update(belief, reader.catalogue.questions[5].key, "unsure")
    assert np.array_equal(before, belief.probs)


def test_accuracy_target_holds_inside_the_design_envelope():
    """The headline claim: first guess right ≥ 95% at up to 10% wrong answers."""
    players = [p for p in PLAYERS if p.label in {"honest", "5% wrong", "10% wrong"}]
    for row in forward_table(players, runs=4, seed=7):
        assert row["first_guess"] >= 0.95, row
        assert row["max_questions"] <= settings.max_questions


def test_redesign_removed_the_collisions():
    assert bank_summary(legacy=True)["collision_groups"] > 0
    assert bank_summary()["collision_groups"] == 0


# --------------------------------------------------------------------------- #
# the scikit-learn decision tree
# --------------------------------------------------------------------------- #
def test_every_tree_isolates_every_dish():
    cat = get_catalogue()
    assert len(forest()) == settings.hint_trees
    for tree in forest():
        stats = tree.stats()
        assert stats["leaves"] == len(cat.dishes)
        assert stats["training_accuracy"] == 1.0
        # Entropy splits should land close to a perfectly balanced tree.
        assert stats["mean_depth"] < stats["ideal_depth"] + 1.0


def test_tree_paths_are_true_and_narrow_to_one():
    table = question_table()
    tree = canonical_tree()
    for name in ("Mole Poblano", "Garlic Naan", "Bigos"):
        column = table.name_index[name]
        path = tree.path(name)
        assert path[0].dishes_before == len(table.names)
        assert path[-1].dishes_after == 1
        for earlier, later in zip(path, path[1:]):
            assert earlier.dishes_after == later.dishes_before
        for split in path:
            assert bool(table.truth[table.key_index[split.key], column]) == split.truth


def test_hard_tree_follows_honest_answers_to_the_dish():
    table = question_table()
    tree = canonical_tree()
    column = table.name_index["Tahchin"]
    reached, used = tree.classify(lambda key: bool(table.truth[table.key_index[key], column]))
    assert reached == "Tahchin"
    assert used == tree.depth("Tahchin")


# --------------------------------------------------------------------------- #
# reverse mode: the hint ladder
# --------------------------------------------------------------------------- #
def test_hint_plans_are_bracketed_by_written_clues_and_never_leak():
    cat = get_catalogue()
    table = question_table()
    pool = cat.names
    columns = hints._columns(pool)
    for tree_index in range(len(forest())):
        for dish in cat.dishes[::7]:
            plan = hints.build_plan(dish, pool, tree_index)
            assert plan[0] == {"kind": "clue", "index": 0, "text": dish["clues"][0]}
            assert plan[-1]["text"] == dish["clues"][3], "the near-giveaway comes last"
            assert [s["text"] for s in plan if s["kind"] == "clue"] == dish["clues"]

            forbidden = leak_terms(dish, include_aliases=True)
            column = table.name_index[dish["name"]]
            alive = np.ones(len(columns), dtype=bool)
            for position, step in enumerate(s for s in plan if s["kind"] == "trait"):
                assert bool(table.truth[table.key_index[step["key"]], column]) == step["truth"]
                assert not find_leak(step["text"], forbidden, dish["name"])
                narrowed = alive & hints._fits(step["key"], step["truth"], columns)
                assert narrowed.sum() < alive.sum(), "every trait must rule something out"
                alive = narrowed
                if position < 4:
                    assert alive.sum() > 1, "no early trait may pin the dish down"


def test_wrong_guess_gets_the_separating_hint():
    cat = get_catalogue()
    table = question_table()
    dish = cat.get("Garlic Naan")
    state = hints.new_state(dish, cat.names, random.Random(1))

    # Butter Chicken is nothing like a flatbread: once a trait is out that it
    # breaks, the chef points back at that trait instead of revealing more.
    state, _ = hints.advance(state)
    while not any(hints._differs(s, "Butter Chicken") for s in hints.revealed_steps(state)
                  if s["kind"] == "trait"):
        state, _ = hints.advance(state)
    state, feedback = hints.judge_wrong_guess(state, "Butter Chicken")
    assert feedback and "Butter Chicken" in feedback
    assert state["contrast"] is None

    # A guess that fits everything so far is ruled out by the very next hint.
    fresh = hints.new_state(dish, cat.names, random.Random(1))
    fresh, feedback = hints.judge_wrong_guess(fresh, "Tandoori Roti")
    assert feedback is None and fresh["contrast"] == "Tandoori Roti"
    fresh, hint = hints.advance(fresh)
    assert hint["kind"] == "trait" and hint["contrast"] == "Tandoori Roti"
    other = table.name_index["Tandoori Roti"]
    assert bool(table.truth[table.key_index[hint["key"]], other]) != hint["truth"]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["dishes"] >= 100


def test_clue_game_full_round(client):
    start = client.post("/api/game/clue/new", json={"difficulty_min": 1, "difficulty_max": 5})
    assert start.status_code == 200
    state = start.json()
    sid = state["session_id"]
    assert state["clue"] and state["hint_kind"] == "clue"
    assert state["hints_revealed"] == 1 and state["clues_revealed"] == 1
    total = state["candidates_left"]

    wrong = client.post("/api/game/clue/guess", json={"session_id": sid, "guess": "zzzz"}).json()
    assert wrong["correct"] is False
    assert wrong["state"]["wrong_guesses"] == 1
    assert wrong["state"]["potential_score"] < state["potential_score"]

    nxt = client.post("/api/game/clue/next", json={"session_id": sid}).json()
    assert nxt["hint_kind"] == "trait", "the ladder follows the opening clue"
    assert nxt["hints_revealed"] == 2 and nxt["hints_used"] == 1 and nxt["clues_revealed"] == 1
    assert nxt["candidates_left"] < total
    assert [h["text"] for h in nxt["history"]] == [state["clue"], nxt["clue"]]

    # Walk the ladder to the end: the field only ever narrows, and the last
    # hint is the dish's own near-giveaway clue.
    left, last = nxt["candidates_left"], nxt
    for _ in range(nxt["hints_total"]):
        step = client.post("/api/game/clue/next", json={"session_id": sid}).json()
        if step["clue"] is None:
            break
        assert step["candidates_left"] <= left
        left, last = step["candidates_left"], step
    assert last["hints_revealed"] == last["hints_total"]
    assert last["hint_kind"] == "clue" and last["clues_revealed"] == 4
    assert last["potential_score"] >= 10

    reveal = client.post("/api/game/clue/giveup", json={"session_id": sid}).json()
    assert reveal["dish"]["name"] == _secret(sid)
    assert reveal["dish"]["emoji"]
    assert len(reveal["all_clues"]) == 4
    assert last["clue"] == reveal["all_clues"][3]


def test_clue_game_win_is_scored(client):
    sid = client.post("/api/game/clue/new", json={}).json()["session_id"]
    result = client.post(
        "/api/game/clue/guess", json={"session_id": sid, "guess": _secret(sid)}
    ).json()
    assert result["correct"] is True
    assert result["score_awarded"] > 0
    assert result["reveal"]["name"] == _secret(sid)
    again = client.post("/api/game/clue/next", json={"session_id": sid})
    assert again.status_code == 400, "a finished round gives no more hints"


def test_clue_game_contrasts_a_wrong_guess(client):
    cat = get_catalogue()
    table = question_table()
    sid = client.post("/api/game/clue/new", json={}).json()["session_id"]
    secret = _secret(sid)
    # Guess a dish that the first trait on the ladder would rule out.
    with SessionLocal() as db:
        plan = db.get(GameSession, sid).hint_state["plan"]
    first_trait = next(step for step in plan if step["kind"] == "trait")
    other = next(n for n in cat.names if n != secret and hints._differs(first_trait, n))

    result = client.post("/api/game/clue/guess", json={"session_id": sid, "guess": other}).json()
    assert result["correct"] is False and result["matched"] == other
    assert result["feedback"] is None, "only the opening clue is out; nothing to contradict yet"

    hint = client.post("/api/game/clue/next", json={"session_id": sid}).json()
    assert hint["clue"].startswith(f"Unlike {other}: ")
    # The statement is true of the hidden dish and false of the guess.
    with SessionLocal() as db:
        state = db.get(GameSession, sid).hint_state
    step = state["plan"][state["revealed"][-1]]
    row = table.truth[table.key_index[step["key"]]]
    assert bool(row[table.name_index[secret]]) == step["truth"]
    assert bool(row[table.name_index[other]]) != step["truth"]

    again = client.post("/api/game/clue/guess", json={"session_id": sid, "guess": other}).json()
    assert again["feedback"] and other in again["feedback"]


def _answer_for(dish: dict, key: str, lie: bool = False) -> str:
    truth = get_catalogue().question_by_key[key].predicate(dish)
    return "yes" if truth != lie else "no"


def test_mind_game_reads_an_honest_mind(client):
    dish = get_catalogue().get("Mohinga")
    state = client.post("/api/game/mind/new", json={}).json()
    sid = state["session_id"]
    assert state["question"] and state["effective_candidates"] > 300
    while state.get("question"):
        state = client.post(
            "/api/game/mind/answer",
            json={"session_id": sid, "key": state["question"]["key"],
                  "answer": _answer_for(dish, state["question"]["key"])},
        ).json()
    assert state["guess"]["name"] == "Mohinga"
    assert state["confidence"] >= settings.guess_threshold or state["questions_asked"] > 8
    assert 0.0 <= state["progress"] <= 1.0
    assert len(state["top_candidates"]) == 5

    done = client.post("/api/game/mind/verdict", json={"session_id": sid, "correct": True}).json()
    assert done["status"] == "finished"
    with SessionLocal() as db:
        row = db.get(GameSession, sid)
        assert row.outcome == "engine_won" and row.secret_dish == "Mohinga"


def test_mind_game_rejects_unknown_question_key(client):
    sid = client.post("/api/game/mind/new", json={}).json()["session_id"]
    r = client.post(
        "/api/game/mind/answer", json={"session_id": sid, "key": "flag:nonsense", "answer": "yes"}
    )
    assert r.status_code == 400


def test_mind_game_only_accepts_the_open_question(client):
    state = client.post("/api/game/mind/new", json={}).json()
    sid, key = state["session_id"], state["question"]["key"]
    other = next(q.key for q in get_catalogue().questions if q.key != key)
    assert client.post("/api/game/mind/answer",
                       json={"session_id": sid, "key": other, "answer": "yes"}).status_code == 400
    assert client.post("/api/game/mind/verdict",
                       json={"session_id": sid, "correct": False}).status_code == 400
    ok = client.post("/api/game/mind/answer", json={"session_id": sid, "key": key, "answer": "yes"})
    assert ok.status_code == 200
    # The same answer sent twice must not be counted twice.
    assert client.post("/api/game/mind/answer",
                       json={"session_id": sid, "key": key, "answer": "yes"}).status_code == 400


def test_mind_game_gives_up_after_three_wrong_guesses_and_reviews_answers(client):
    dish = get_catalogue().get("Poutine")
    state = client.post("/api/game/mind/new", json={}).json()
    sid = state["session_id"]
    lied = 0
    for _ in range(200):
        if state["status"] != "active":
            break
        if state.get("question"):
            lie = lied < 2          # mislead it twice, then play straight
            lied += lie
            state = client.post(
                "/api/game/mind/answer",
                json={"session_id": sid, "key": state["question"]["key"],
                      "answer": _answer_for(dish, state["question"]["key"], lie)},
            ).json()
        else:
            # Whatever it names, deny it.
            state = client.post("/api/game/mind/verdict",
                                json={"session_id": sid, "correct": False}).json()
    assert state["status"] == "finished" and state["guess"] is None
    assert state["questions_asked"] <= settings.max_questions

    told = client.post("/api/game/mind/reveal", json={"session_id": sid, "actual": "poutine"}).json()
    assert told["revealed"]["name"] == "Poutine"
    assert len(told["review"]) == 2, "exactly the two lies should be flagged"
    assert all(row["answered"] in {"yes", "no"} for row in told["review"])

    noise = client.get("/api/analytics/answer-noise").json()
    assert noise["rounds"] >= 1
    assert set(noise["assumed"]) == {"attribute", "origin", "cuisine", "ingredient"}
    assert sum(row["answers"] for row in noise["observed"].values()) >= told["questions_asked"]


def test_mind_reveal_of_an_unknown_dish_is_accepted(client):
    sid = client.post("/api/game/mind/new", json={}).json()["session_id"]
    told = client.post("/api/game/mind/reveal",
                       json={"session_id": sid, "actual": "qwertyuiop asdfgh"}).json()
    assert told["status"] == "finished" and told["revealed"] is None and told["review"] == []


def test_predict_and_feedback(client):
    r = client.post(
        "/api/predict",
        json={"text": "soured cabbage stewed with smoked sausage, mushrooms and prunes"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"]
    assert body["predictions"][0]["dish"] == "Bigos"
    assert body["latency_ms"] >= 0

    fb = client.post(
        "/api/predict/feedback",
        json={
            "prediction_id": body["prediction_id"],
            "input_text": "soured cabbage stew",
            "predicted_label": body["predictions"][0]["dish"],
            "correct_label": "Bigos",
        },
    ).json()
    assert fb["ok"] and fb["was_correct"]


def test_feedback_rejects_unknown_dish(client):
    r = client.post(
        "/api/predict/feedback",
        json={"input_text": "x", "correct_label": "Not A Real Dish"},
    )
    assert r.status_code == 400


def test_analytics_shape(client):
    body = client.get("/api/analytics").json()
    assert body["games_played"] >= 1
    assert "hardest_dishes" in body
    assert 0.0 <= body["win_rate"] <= 1.0


def test_unknown_session_is_404(client):
    r = client.post("/api/game/clue/guess", json={"session_id": "deadbeef", "guess": "x"})
    assert r.status_code == 404


def test_tree_endpoints(client):
    body = client.get("/api/tree").json()
    assert body["model"] == "sklearn.tree.DecisionTreeClassifier"
    assert body["stats"]["leaves"] == len(get_catalogue().dishes)
    assert {"question", "yes", "no", "dishes"} <= set(body["tree"])
    text = client.get("/api/tree/text").text
    assert text.count("→") == len(get_catalogue().dishes)
    engine = client.get("/api/tree/engine").json()
    assert engine["forward"]["guess_threshold"] == settings.guess_threshold


def test_front_end_is_served_and_old_pages_redirect_into_it(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'src="app.js"' in page.text and 'src="api.js"' in page.text
    script = client.get("/app.js").text
    for endpoint in ("/api/game/mind/new", "/api/game/mind/answer", "/api/game/mind/verdict",
                     "/api/game/mind/reveal", "/api/game/clue/new", "/api/game/clue/next",
                     "/api/game/clue/guess", "/api/game/clue/giveup"):
        assert endpoint in script, f"the app never calls {endpoint}"
    assert client.get("/styles.css").status_code == 200
    assert client.get("/tree").status_code == 200

    for old, mode in (("chef-guess.html", "forward"), ("you-guess.html", "reverse")):
        response = client.get(f"/data/{old}", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == f"/#{mode}"
    # Nothing else in that folder (notebooks, raw datasets) is reachable.
    assert client.get("/data/Reviews.csv").status_code == 404
    assert client.get("/data/..%2Ffood-guess-game%2Fapp%2Fconfig.py").status_code == 404
    # API routes still win over the front-end mount.
    assert client.get("/api/health").json()["status"] == "ok"
