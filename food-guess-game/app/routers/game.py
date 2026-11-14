"""Game endpoints for both play modes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.db_models import DishStat, GameSession, Guess, Turn
from app.schemas import (
    AnswerIn,
    Candidate,
    ClueGameState,
    DishCard,
    GuessIn,
    GuessOut,
    MindQuestion,
    MindState,
    NewClueGame,
    NewMindGame,
    RevealOut,
    VerdictIn,
)
from app.services.catalog import get_catalogue
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
    )


def _log_turn(db: Session, session: GameSession, kind: str, payload: dict) -> None:
    index = db.query(Turn).filter(Turn.session_id == session.id).count()
    db.add(Turn(session_id=session.id, index=index, kind=kind, payload=payload))


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


def _hints_used(db: Session, session: GameSession) -> int:
    return (
        db.query(Turn)
        .filter(Turn.session_id == session.id, Turn.kind == "hint")
        .count()
    )


def _clue_state(
    db: Session, session: GameSession, clue: str | None = None, message: str | None = None
) -> ClueGameState:
    dish = get_catalogue().get(session.secret_dish)
    hints = _hints_used(db, session)
    return ClueGameState(
        session_id=session.id,
        status=session.status,
        clue=clue,
        clues_revealed=session.clues_revealed,
        clues_available=len(dish["clues"]),
        wrong_guesses=session.wrong_guesses,
        hints_used=hints,
        potential_score=clue_score(
            dish["difficulty"], session.clues_revealed, session.wrong_guesses, hints
        ),
        difficulty=dish["difficulty"],
        region_hint=session.region,
        message=message,
    )


# --------------------------------------------------------------------------- #
# Mode 1 — guess the chef's dish
# --------------------------------------------------------------------------- #
@router.post("/clue/new", response_model=ClueGameState)
def new_clue_game(body: NewClueGame, db: Session = Depends(get_db)) -> ClueGameState:
    catalogue = get_catalogue()
    if body.difficulty_min > body.difficulty_max:
        raise HTTPException(400, "difficulty_min must not exceed difficulty_max")

    dish = catalogue.pick(
        body.difficulty_min, body.difficulty_max, body.region, exclude=body.exclude
    )
    session = GameSession(
        client_id=body.client_id,
        mode="clue",
        secret_dish=dish["name"],
        difficulty_min=body.difficulty_min,
        difficulty_max=body.difficulty_max,
        region=body.region,
        status="active",
        clues_revealed=1,
    )
    db.add(session)
    db.flush()

    first_clue = dish["clues"][0]
    _log_turn(db, session, "clue", {"index": 0, "text": first_clue})
    _bump_stat(db, dish["name"], times_selected=1)
    db.commit()

    return _clue_state(db, session, clue=first_clue, message="The chef has chosen a dish.")


@router.post("/clue/next", response_model=ClueGameState)
def next_clue(body: GuessIn | None = None, session_id: str = "", db: Session = Depends(get_db)):
    """Reveal the next written clue, or a generated attribute hint once they run out."""
    sid = session_id or (body.session_id if body else "")
    session = _load(db, sid, "clue")
    if session.status != "active":
        raise HTTPException(400, "this game is already finished")

    dish = get_catalogue().get(session.secret_dish)
    if session.clues_revealed < len(dish["clues"]):
        text = dish["clues"][session.clues_revealed]
        session.clues_revealed += 1
        _log_turn(db, session, "clue", {"index": session.clues_revealed - 1, "text": text})
        db.commit()
        return _clue_state(db, session, clue=text)

    used = [
        t.payload.get("kind")
        for t in db.query(Turn).filter(Turn.session_id == session.id, Turn.kind == "hint")
    ]
    hint = get_catalogue().attribute_hint(dish, [u for u in used if u])
    if hint is None:
        return _clue_state(db, session, message="The chef has nothing left to give away.")

    kind, text = hint
    _log_turn(db, session, "hint", {"kind": kind, "text": text})
    db.commit()
    return _clue_state(db, session, clue=text, message="An extra hint costs you points.")


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
    if correct:
        hints = _hints_used(db, session)
        awarded = clue_score(
            dish["difficulty"], session.clues_revealed, session.wrong_guesses, hints
        )
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
        message = (
            f"Not quite — but you are close to something."
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
        state=_clue_state(db, session, message=message),
        reveal=reveal,
        score_awarded=awarded,
    )


@router.post("/clue/giveup", response_model=RevealOut)
def give_up(body: GuessIn, db: Session = Depends(get_db)) -> RevealOut:
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
        state=_clue_state(db, session, message="The chef reveals all."),
    )


# --------------------------------------------------------------------------- #
# Mode 2 — the chef reads your mind
# --------------------------------------------------------------------------- #
def _mind_state(
    session: GameSession,
    belief: Belief,
    reader: MindReader,
    question=None,
    guess=None,
    message: str | None = None,
) -> MindState:
    return MindState(
        session_id=session.id,
        status=session.status,
        question=MindQuestion(key=question.key, text=question.text) if question else None,
        guess=_card(get_catalogue().get(guess)) if guess else None,
        questions_asked=session.questions_asked,
        max_questions=settings.max_questions,
        confidence=round(belief.confidence, 4),
        progress=round(reader.progress(belief), 4),
        top_candidates=[
            Candidate(dish=n, probability=round(p, 4)) for n, p in belief.top(5)
        ],
        message=message,
    )


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
    )
    db.add(session)
    db.flush()

    question = reader.next_question(belief, [])
    if question is None:
        raise HTTPException(500, "no usable questions for this pool")
    session.asked_keys = [question.key]
    session.questions_asked = 0
    _log_turn(db, session, "question", {"key": question.key, "text": question.text})
    db.commit()

    return _mind_state(
        session, belief, reader, question=question,
        message="Think of a dish and hold it in your mind. Answer honestly.",
    )


@router.post("/mind/answer", response_model=MindState)
def answer_question(body: AnswerIn, db: Session = Depends(get_db)) -> MindState:
    session = _load(db, body.session_id, "mind")
    if session.status != "active":
        raise HTTPException(400, "this game is already finished")

    catalogue = get_catalogue()
    reader = MindReader(catalogue)
    if body.key not in catalogue.question_by_key:
        raise HTTPException(400, f"unknown question key {body.key!r}")

    belief = Belief.from_dict(session.belief or {})
    belief = reader.update(belief, body.key, body.answer)
    session.questions_asked += 1
    _log_turn(db, session, "answer", {"key": body.key, "answer": body.answer})

    asked = list(session.asked_keys or [])
    if body.key not in asked:
        asked.append(body.key)

    if reader.should_guess(belief, session.questions_asked):
        candidate = next(
            (n for n, _ in belief.top(len(belief.names))
             if n not in set(session.rejected or [])),
            belief.top(1)[0][0],
        )
        session.belief = belief.to_dict()
        session.asked_keys = asked
        _log_turn(db, session, "engine_guess", {"dish": candidate,
                                                "confidence": belief.confidence})
        db.commit()
        return _mind_state(
            session, belief, reader, guess=candidate,
            message="I think I have it.",
        )

    question = reader.next_question(belief, asked)
    if question is None:
        candidate = belief.top(1)[0][0]
        session.belief = belief.to_dict()
        session.asked_keys = asked
        db.commit()
        return _mind_state(session, belief, reader, guess=candidate,
                           message="I have run out of useful questions.")

    asked.append(question.key)
    session.belief = belief.to_dict()
    session.asked_keys = asked
    _log_turn(db, session, "question", {"key": question.key, "text": question.text})
    db.commit()
    return _mind_state(session, belief, reader, question=question)


@router.post("/mind/verdict", response_model=MindState)
def mind_verdict(body: VerdictIn, db: Session = Depends(get_db)) -> MindState:
    """The player confirms or rejects the engine's guess."""
    session = _load(db, body.session_id, "mind")
    catalogue = get_catalogue()
    reader = MindReader(catalogue)
    belief = Belief.from_dict(session.belief or {})

    last_guess = (
        db.query(Turn)
        .filter(Turn.session_id == session.id, Turn.kind == "engine_guess")
        .order_by(Turn.index.desc())
        .first()
    )
    guessed = last_guess.payload.get("dish") if last_guess else None

    if body.correct:
        session.status = "finished"
        session.outcome = "engine_won"
        session.secret_dish = guessed
        session.finished_at = _now()
        session.belief = belief.to_dict()
        if guessed:
            _bump_stat(db, guessed, times_selected=1, times_solved=1,
                       total_questions=session.questions_asked)
        _log_turn(db, session, "verdict", {"correct": True, "dish": guessed})
        db.commit()
        return _mind_state(session, belief, reader, guess=guessed,
                           message="Read your mind in "
                                   f"{session.questions_asked} questions.")

    rejected = list(session.rejected or [])
    if guessed:
        rejected.append(guessed)
        belief = reader.reject(belief, guessed)
    session.rejected = rejected
    session.belief = belief.to_dict()
    _log_turn(db, session, "verdict", {"correct": False, "dish": guessed,
                                       "actual": body.actual})

    if body.actual:
        session.status = "finished"
        session.outcome = "player_won"
        session.secret_dish = body.actual
        session.finished_at = _now()
        _bump_stat(db, body.actual, times_selected=1,
                   total_questions=session.questions_asked)
        db.commit()
        return _mind_state(session, belief, reader,
                           message=f"You win. {body.actual} it is — noted for next time.")

    if session.questions_asked >= settings.max_questions:
        session.status = "finished"
        session.outcome = "player_won"
        session.finished_at = _now()
        db.commit()
        return _mind_state(session, belief, reader, message="You beat me. What was it?")

    question = reader.next_question(belief, list(session.asked_keys or []))
    if question is None:
        session.status = "finished"
        session.outcome = "player_won"
        session.finished_at = _now()
        db.commit()
        return _mind_state(session, belief, reader, message="I am out of questions. You win.")

    asked = list(session.asked_keys or [])
    asked.append(question.key)
    session.asked_keys = asked
    _log_turn(db, session, "question", {"key": question.key, "text": question.text})
    db.commit()
    return _mind_state(session, belief, reader, question=question,
                       message="Let me narrow it down further.")


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
