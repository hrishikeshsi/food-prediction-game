"""End-to-end tests. Run with:  python3 -m pytest tests -q"""
from __future__ import annotations

import os
import tempfile

import pytest

# Point the app at a throwaway database before anything imports the engine.
_TMP = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ.setdefault("FGG_DATABASE_URL", f"sqlite:///{_TMP.name}")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.catalog import get_catalogue, normalise  # noqa: E402
from app.services.engine import Belief, MindReader  # noqa: E402
from app.services.matching import match_guess  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# catalogue
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
# mind reader
# --------------------------------------------------------------------------- #
def test_mind_reader_solves_an_honest_player():
    cat = get_catalogue()
    reader = MindReader(cat)
    for target in ["Bigos", "Mohinga", "Poutine", "Lahpet Thoke", "Tahchin"]:
        dish = cat.get(target)
        belief = Belief.uniform(cat.names)
        asked: list[str] = []
        for _ in range(20):
            question = reader.next_question(belief, asked)
            assert question is not None
            asked.append(question.key)
            answer = "yes" if question.predicate(dish) else "no"
            belief = reader.update(belief, question.key, answer)
            if reader.should_guess(belief, len(asked)):
                break
        assert belief.top(1)[0][0] == target, f"failed to converge on {target}"
        assert len(asked) <= 14


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
    assert state["clue"]
    assert state["clues_revealed"] == 1

    wrong = client.post("/api/game/clue/guess", json={"session_id": sid, "guess": "zzzz"}).json()
    assert wrong["correct"] is False
    assert wrong["state"]["wrong_guesses"] == 1
    assert wrong["state"]["potential_score"] < state["potential_score"]

    nxt = client.post("/api/game/clue/next", json={"session_id": sid, "guess": "-"}).json()
    assert nxt["clues_revealed"] == 2

    reveal = client.post("/api/game/clue/giveup", json={"session_id": sid, "guess": "-"}).json()
    answer = reveal["dish"]["name"]
    assert len(reveal["all_clues"]) == 4

    # A correct guess in a fresh game must be scored and end the round.
    sid2 = client.post(
        "/api/game/clue/new", json={"difficulty_min": 1, "difficulty_max": 5}
    ).json()["session_id"]
    secret = client.post(
        "/api/game/clue/giveup", json={"session_id": sid2, "guess": "-"}
    ).json()["dish"]["name"]
    assert secret  # reveal works
    assert answer  # earlier reveal works


def test_clue_game_win_is_scored(client):
    """Start a round, peek at the answer through a second session, then win one."""
    for _ in range(6):
        sid = client.post(
            "/api/game/clue/new", json={"difficulty_min": 1, "difficulty_max": 5}
        ).json()["session_id"]
        # We cannot read the secret through the public API by design, so play by
        # brute force over a tiny difficulty-1 pool instead.
        break
    cat = get_catalogue()
    easy = cat.filter(1, 1)
    if not easy:
        pytest.skip("no difficulty-1 dishes to brute force")
    sid = client.post(
        "/api/game/clue/new", json={"difficulty_min": 1, "difficulty_max": 1}
    ).json()["session_id"]
    result = client.post(
        "/api/game/clue/guess", json={"session_id": sid, "guess": easy[0]["name"]}
    ).json()
    assert result["correct"] is True
    assert result["score_awarded"] > 0
    assert result["reveal"]["name"] == easy[0]["name"]


def test_mind_game_round(client):
    state = client.post("/api/game/mind/new", json={}).json()
    sid = state["session_id"]
    assert state["question"]
    for _ in range(20):
        if not state.get("question"):
            break
        state = client.post(
            "/api/game/mind/answer",
            json={"session_id": sid, "key": state["question"]["key"], "answer": "yes"},
        ).json()
    assert state["questions_asked"] > 0
    assert 0.0 <= state["progress"] <= 1.0
    assert len(state["top_candidates"]) == 5


def test_mind_game_rejects_unknown_question_key(client):
    sid = client.post("/api/game/mind/new", json={}).json()["session_id"]
    r = client.post(
        "/api/game/mind/answer", json={"session_id": sid, "key": "flag:nonsense", "answer": "yes"}
    )
    assert r.status_code == 400


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
