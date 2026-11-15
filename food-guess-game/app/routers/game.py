"""Game endpoints for both play modes.

``/clue/*``  reverse mode — the chef hides a dish and reveals hints.
``/mind/*``  forward mode — the player hides a dish and the engine asks.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import DishStat, GameSession, Guess, Turn
from app.schemas import (
    AnswerIn,
    Candidate,
    ClueGameState,
    Disagreement,
    DishCard,
    GuessIn,
    GuessOut,
    Hint,
    MindQuestion,
    MindState,
    NewClueGame,
    NewMindGame,
    RevealIn,
    RevealOut,
    SessionRef,
    VerdictIn,
)
from app.services import hints
from app.services.catalog import dish_emoji, get_catalogue
from app.services.engine import Belief, MindReader, build_initial_belief, clue_score
from app.services.inference import get_predictor
from app.services.matching import match_guess

router = APIRouter(prefix="/api/game", tags=["game"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _card(dish: dict) -> DishCard:
    return DishCard(
        name=dish["name"],
        cuisine=dish["cuisine"],
        region=dish["region"],
        country=dish["country"],
        course=dish["course"],
        difficulty=dish["difficulty"],
        key_ingredients=dish["key_ingredients"],
        fact=dish["fact"],
        emoji=dish_emoji(dish),
    )


def _log_turn(db: Session, session: GameSession, kind: str, payload: dict) -> None:
    index = db.query(Turn).filter(Turn.session_id == session.id).count()
    db.add(Turn(session_id=session.id, index=index, kind=kind, payload=payload))
    db.flush()


def _bump_stat(db: Session, dish: str, **increments: int) -> None:
    stat = db.get(DishStat, dish)
    if stat is None:
        stat = DishStat(dish=dish)
        db.add(stat)
        db.flush()
    for field, value in increments.items():
        setattr(stat, field, (getattr(stat, field) or 0) + value)


def _load(db: Session, session_id: str, mode: str) -> GameSession:
    session = db.get(GameSession, session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    if session.mode != mode:
        raise HTTPException(400, f"session {session_id} is a '{session.mode}' game")
    return session


def _pool(difficulty_min: int, difficulty_max: int, region: str | None) -> list[str]:
    """Dishes a round may be about. An empty filter falls back to everything."""
    catalogue = get_catalogue()
    names = [d["name"] for d in catalogue.filter(difficulty_min, difficulty_max, region)]
    return names or catalogue.names


def _session_pool(session: GameSession) -> list[str]:
    return _pool(session.difficulty_min, session.difficulty_max, session.region)


# --------------------------------------------------------------------------- #
# Reverse mode — guess the chef's dish
# --------------------------------------------------------------------------- #
def _clue_state(
    session: GameSession, hint: dict | None = None, message: str | None = None
) -> ClueGameState:
    dish = get_catalogue().get(session.secret_dish)
    state = session.hint_state
    clues, traits = hints.counts(state)
    return ClueGameState(
        session_id=session.id,
        status=session.status,
        clue=hints.hint_text(hint) if hint else None,
        hint_kind=hint["kind"] if hint else None,
        clues_revealed=clues,
        clues_available=len(dish["clues"]),
        hints_used=traits,
        hints_revealed=clues + traits,
        hints_total=len(state["plan"]),
        candidates_left=hints.candidates_left(state, _session_pool(session)),
        wrong_guesses=session.wrong_guesses,
        potential_score=clue_score(dish["difficulty"], clues, session.wrong_guesses, traits),
        difficulty=dish["difficulty"],
        region_hint=session.region,
        message=message,
        history=[Hint(kind=h["kind"], text=h["text"]) for h in state.get("log", [])],
    )


def _record_hint(db: Session, session: GameSession, state: dict, hint: dict) -> None:
    """Persist a revealed hint. ``state`` is reassigned so the JSON column saves."""
    text = hints.hint_text(hint)
    state["log"] = [*state.get("log", []), {"kind": hint["kind"], "text": text}]
    session.hint_state = state
    session.clues_revealed = len(state["revealed"])
    _log_turn(
        db, session, "clue" if hint["kind"] == "clue" else "hint",
        {"text": text, "key": hint.get("key"), "contrast": hint.get("contrast")},
    )


@router.post("/clue/new", response_model=ClueGameState)
def new_clue_game(body: NewClueGame, db: Session = Depends(get_db)) -> ClueGameState:
    catalogue = get_catalogue()
    if body.difficulty_min > body.difficulty_max:
        raise HTTPException(400, "difficulty_min must not exceed difficulty_max")

    dish = catalogue.pick(
        body.difficulty_min, body.difficulty_max, body.region, exclude=body.exclude
    )
    pool = _pool(body.difficulty_min, body.difficulty_max, body.region)
    state = hints.new_state(dish, pool)
    opening = state["plan"][0]

    session = GameSession(
        client_id=body.client_id,
        mode="clue",
        secret_dish=dish["name"],
        difficulty_min=body.difficulty_min,
        difficulty_max=body.difficulty_max,
        region=body.region,
        status="active",
    )
    db.add(session)
    db.flush()
    _record_hint(db, session, state, opening)
    _bump_stat(db, dish["name"], times_selected=1)
    db.commit()

    return _clue_state(session, hint=opening, message="The chef has chosen a dish.")


@router.post("/clue/next", response_model=ClueGameState)
def next_clue(body: SessionRef, db: Session = Depends(get_db)) -> ClueGameState:
    """Reveal the next hint on the ladder."""
    session = _load(db, body.session_id, "clue")
    if session.status != "active":
        raise HTTPException(400, "this game is already finished")

    state, hint = hints.advance(session.hint_state)
    if hint is None:
        session.hint_state = state
        db.commit()
        return _clue_state(session, message="The chef has nothing left to give away.")

    _record_hint(db, session, state, hint)
    db.commit()
    message = None
    if hint["kind"] == "trait":
        message = (
            "That rules your guess out." if hint.get("contrast")
            else "A trait narrows the field; it costs fewer points than a clue."
        )
    return _clue_state(session, hint=hint, message=message)


@router.post("/clue/guess", response_model=GuessOut)
def submit_guess(body: GuessIn, db: Session = Depends(get_db)) -> GuessOut:
    session = _load(db, body.session_id, "clue")
    if session.status != "active":
        raise HTTPException(400, "this game is already finished")

    catalogue = get_catalogue()
    dish = catalogue.get(session.secret_dish)
    result = match_guess(body.guess, catalogue)
    correct = result.accepted and result.dish == session.secret_dish

    db.add(
        Guess(
            session_id=session.id,
            raw_text=body.guess,
            matched_dish=result.dish if result.accepted else None,
            match_score=result.score,
            correct=correct,
        )
    )
    _log_turn(
        db,
        session,
        "guess",
        {"text": body.guess, "matched": result.dish, "score": result.score, "correct": correct},
    )

    awarded = 0
    reveal = None
    feedback = None
    if correct:
        clues, traits = hints.counts(session.hint_state)
        awarded = clue_score(dish["difficulty"], clues, session.wrong_guesses, traits)
        session.status = "finished"
        session.outcome = "solved"
        session.score = awarded
        session.finished_at = _now()
        reveal = _card(dish)
        _bump_stat(
            db,
            dish["name"],
            times_solved=1,
            total_clues=session.clues_revealed,
            total_wrong_guesses=session.wrong_guesses,
        )
        message = "Correct. The chef is impressed."
    else:
        session.wrong_guesses += 1
        if result.accepted:
            # A real dish, just not this one: either it breaks a hint already
            # given, or the next hint is chosen to rule it out.
            session.hint_state, feedback = hints.judge_wrong_guess(
                session.hint_state, result.dish
            )
        message = feedback or (
            "Not quite — but you are close to something."
            if result.dish and result.score > 0.6
            else "No. Try again, or take another clue."
        )

    db.commit()
    return GuessOut(
        correct=correct,
        matched=result.dish if result.accepted else None,
        match_score=result.score,
        match_method=result.method,
        near_miss=(result.dish if (not correct and 0.6 <= result.score < 0.82) else None),
        feedback=feedback,
        state=_clue_state(session, message=message),
        reveal=reveal,
        score_awarded=awarded,
    )


@router.post("/clue/giveup", response_model=RevealOut)
def give_up(body: SessionRef, db: Session = Depends(get_db)) -> RevealOut:
    session = _load(db, body.session_id, "clue")
    dish = get_catalogue().get(session.secret_dish)
    if session.status == "active":
        session.status = "finished"
        session.outcome = "gave_up"
        session.score = 0
        session.finished_at = _now()
        _bump_stat(db, dish["name"], total_clues=session.clues_revealed)
        _log_turn(db, session, "giveup", {"dish": dish["name"]})
        db.commit()
    return RevealOut(
        dish=_card(dish),
        all_clues=dish["clues"],
        state=_clue_state(session, message="The chef reveals all."),
    )


# --------------------------------------------------------------------------- #
# Forward mode — the chef reads your mind
# --------------------------------------------------------------------------- #
def _mind_state(
    session: GameSession,
    belief: Belief,
    reader: MindReader,
    question=None,
    guess: str | None = None,
    message: str | None = None,
    revealed: str | None = None,
    review: list[dict] | None = None,
) -> MindState:
    catalogue = get_catalogue()
    return MindState(
        session_id=session.id,
        status=session.status,
        question=MindQuestion(key=question.key, text=question.text) if question else None,
        guess=_card(catalogue.get(guess)) if guess else None,
        questions_asked=session.questions_asked,
        max_questions=reader.policy.max_questions,
        guesses_made=len(session.rejected or []) + (1 if guess else 0),
        max_guesses=reader.policy.max_guesses,
        confidence=round(belief.confidence, 4),
        progress=round(reader.progress(belief), 4),
        effective_candidates=round(reader.effective_candidates(belief), 2),
        top_candidates=[
            Candidate(dish=n, probability=round(p, 4)) for n, p in belief.top(5)
        ],
        message=message,
        revealed=_card(catalogue.get(revealed)) if revealed else None,
        review=[Disagreement(**row) for row in review] if review is not None else None,
    )


def _pending_question(session: GameSession) -> str | None:
    """The question asked but not yet answered, if there is one."""
    asked = session.asked_keys or []
    return asked[session.questions_asked] if len(asked) > session.questions_asked else None


def _pending_guess(db: Session, session: GameSession) -> str | None:
    """The engine's guess awaiting a verdict, if there is one."""
    last = (
        db.query(Turn)
        .filter(Turn.session_id == session.id,
                Turn.kind.in_(("question", "engine_guess", "verdict")))
        .order_by(Turn.index.desc())
        .first()
    )
    return last.payload.get("dish") if last and last.kind == "engine_guess" else None


def _advance_mind(
    db: Session, session: GameSession, belief: Belief, reader: MindReader,
    message: str | None = None,
) -> MindState:
    """Let the engine make its next move — ask or guess — and persist it."""
    asked = list(session.asked_keys or [])
    move, what = reader.next_move(
        belief, asked, session.questions_asked, session.rejected or []
    )
    session.belief = belief.to_dict()

    if move == "ask":
        session.asked_keys = [*asked, what.key]
        _log_turn(db, session, "question", {"key": what.key, "text": what.text})
        db.commit()
        return _mind_state(session, belief, reader, question=what, message=message)

    confidence = belief.to_dict().get(what, 0.0)
    _log_turn(db, session, "engine_guess", {"dish": what, "confidence": confidence})
    db.commit()
    return _mind_state(session, belief, reader, guess=what,
                       message=message or "I think I have it.")


def _finish_with_reveal(
    db: Session, session: GameSession, belief: Belief, reader: MindReader, actual: str
) -> MindState:
    """The player says what the dish was. Record it and show where we disagreed."""
    catalogue = get_catalogue()
    match = match_guess(actual, catalogue)
    name = match.dish if match.accepted else None

    if session.status == "active":
        session.status = "finished"
        session.outcome = "player_won"
        session.finished_at = _now()
    session.secret_dish = name or actual[:120]
    session.belief = belief.to_dict()
    _log_turn(db, session, "reveal", {"text": actual, "dish": name})

    if name is None:
        db.commit()
        return _mind_state(
            session, belief, reader, review=[],
            message=(f"“{actual}” is not in my book yet — I know "
                     f"{len(catalogue.dishes)} dishes. You win this one."),
        )

    answers = [
        (t.payload.get("key"), t.payload.get("answer"))
        for t in db.query(Turn)
        .filter(Turn.session_id == session.id, Turn.kind == "answer")
        .order_by(Turn.index)
    ]
    review = reader.disagreements(name, answers)
    _bump_stat(db, name, times_selected=1, total_questions=session.questions_asked)
    db.commit()

    if review:
        message = (f"{name} — you win. We disagreed on {len(review)} "
                   f"answer{'s' if len(review) != 1 else ''}; that is what threw me.")
    else:
        message = f"{name} — you win. Your answers matched my book; I just ran out of road."
    return _mind_state(session, belief, reader, message=message, revealed=name, review=review)


@router.post("/mind/new", response_model=MindState)
def new_mind_game(body: NewMindGame, db: Session = Depends(get_db)) -> MindState:
    catalogue = get_catalogue()
    pool = catalogue.filter(body.difficulty_min, body.difficulty_max, body.region)
    if not pool:
        raise HTTPException(400, "no dishes match those filters")
    names = [d["name"] for d in pool]

    prior = None
    if body.hint:
        distribution = get_predictor().distribution(body.hint)
        prior = {n: distribution.get(n, 0.0) for n in names}

    belief = build_initial_belief(names, prior)
    reader = MindReader(catalogue)

    session = GameSession(
        client_id=body.client_id,
        mode="mind",
        difficulty_min=body.difficulty_min,
        difficulty_max=body.difficulty_max,
        region=body.region,
        status="active",
        belief=belief.to_dict(),
        asked_keys=[],
        rejected=[],
        questions_asked=0,
    )
    db.add(session)
    db.flush()

    return _advance_mind(
        db, session, belief, reader,
        message="Think of a dish and hold it in your mind. Answer honestly.",
    )


@router.post("/mind/answer", response_model=MindState)
def answer_question(body: AnswerIn, db: Session = Depends(get_db)) -> MindState:
    session = _load(db, body.session_id, "mind")
    if session.status != "active":
        raise HTTPException(400, "this game is already finished")

    catalogue = get_catalogue()
    if body.key not in catalogue.question_by_key:
        raise HTTPException(400, f"unknown question key {body.key!r}")
    pending = _pending_question(session)
    if body.key != pending:
        # Answering anything but the open question would update the belief
        # twice for one piece of evidence, or for evidence never asked for.
        raise HTTPException(
            400,
            "that is not the question I asked" if pending
            else "there is no open question — confirm or reject my guess",
        )

    reader = MindReader(catalogue)
    belief = Belief.from_dict(session.belief or {})
    belief = reader.update(belief, body.key, body.answer)
    session.questions_asked += 1
    _log_turn(db, session, "answer", {"key": body.key, "answer": body.answer})
    return _advance_mind(db, session, belief, reader)


@router.post("/mind/verdict", response_model=MindState)
def mind_verdict(body: VerdictIn, db: Session = Depends(get_db)) -> MindState:
    """The player confirms or rejects the engine's guess."""
    session = _load(db, body.session_id, "mind")
    reader = MindReader(get_catalogue())
    belief = Belief.from_dict(session.belief or {})

    if session.status != "active":
        if body.actual and session.outcome != "engine_won":
            return _finish_with_reveal(db, session, belief, reader, body.actual)
        raise HTTPException(400, "this game is already finished")

    guessed = _pending_guess(db, session)
    if guessed is None:
        if body.actual and not body.correct:
            return _finish_with_reveal(db, session, belief, reader, body.actual)
        raise HTTPException(400, "there is no guess to judge yet")

    if body.correct:
        session.status = "finished"
        session.outcome = "engine_won"
        session.secret_dish = guessed
        session.finished_at = _now()
        _bump_stat(db, guessed, times_selected=1, times_solved=1,
                   total_questions=session.questions_asked)
        _log_turn(db, session, "verdict", {"correct": True, "dish": guessed})
        db.commit()
        plural = "" if session.questions_asked == 1 else "s"
        return _mind_state(
            session, belief, reader, guess=guessed,
            message=f"Read your mind in {session.questions_asked} question{plural}.",
        )

    rejected = [*(session.rejected or []), guessed]
    belief = reader.reject(belief, guessed)
    session.rejected = rejected
    session.belief = belief.to_dict()
    _log_turn(db, session, "verdict", {"correct": False, "dish": guessed,
                                       "actual": body.actual})

    if body.actual:
        return _finish_with_reveal(db, session, belief, reader, body.actual)

    if len(rejected) >= reader.policy.max_guesses:
        session.status = "finished"
        session.outcome = "player_won"
        session.finished_at = _now()
        db.commit()
        return _mind_state(session, belief, reader, message="You beat me. What was it?")

    return _advance_mind(db, session, belief, reader,
                         message="Let me narrow it down further.")


@router.post("/mind/reveal", response_model=MindState)
def mind_reveal(body: RevealIn, db: Session = Depends(get_db)) -> MindState:
    """The player gives up, or tells a beaten engine what the dish was."""
    session = _load(db, body.session_id, "mind")
    if session.outcome == "engine_won":
        raise HTTPException(400, "the engine already got this one right")
    reader = MindReader(get_catalogue())
    belief = Belief.from_dict(session.belief or {})
    return _finish_with_reveal(db, session, belief, reader, body.actual)


# --------------------------------------------------------------------------- #
# Catalogue browsing
# --------------------------------------------------------------------------- #
@router.get("/regions")
def regions() -> dict:
    from app.services.catalog import REGION_GROUPS

    catalogue = get_catalogue()
    return {
        "groups": sorted(REGION_GROUPS),
        "regions": catalogue.regions(),
        "total_dishes": len(catalogue.dishes),
    }
