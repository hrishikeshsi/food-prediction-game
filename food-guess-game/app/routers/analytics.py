"""Aggregate statistics over everything the database has recorded."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.db_models import DishStat, Feedback, GameSession, Prediction, Turn
from app.schemas import AnalyticsOut
from app.services.catalog import get_catalogue
from app.services.engine import question_table

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _round(value, digits: int = 3):
    return None if value is None else round(float(value), digits)


@router.get("", response_model=AnalyticsOut)
def analytics(db: Session = Depends(get_db)) -> AnalyticsOut:
    finished = select(GameSession).where(GameSession.status == "finished").subquery()

    games_played = db.scalar(select(func.count()).select_from(finished)) or 0
    games_won = (
        db.scalar(
            select(func.count()).select_from(finished).where(
                finished.c.outcome.in_(("solved", "player_won"))
            )
        )
        or 0
    )

    avg_clues = db.scalar(
        select(func.avg(finished.c.clues_revealed)).where(finished.c.outcome == "solved")
    )
    avg_questions = db.scalar(
        select(func.avg(finished.c.questions_asked)).where(finished.c.outcome == "engine_won")
    )

    mind_total = (
        db.scalar(
            select(func.count()).select_from(finished).where(
                finished.c.mode == "mind"
            )
        )
        or 0
    )
    mind_won = (
        db.scalar(
            select(func.count()).select_from(finished).where(
                finished.c.mode == "mind", finished.c.outcome == "engine_won"
            )
        )
        or 0
    )

    feedback_total = db.scalar(select(func.count()).select_from(Feedback)) or 0
    feedback_right = (
        db.scalar(select(func.count()).select_from(Feedback).where(Feedback.was_correct))
        or 0
    )

    stats = db.scalars(select(DishStat).where(DishStat.times_selected > 0)).all()
    scored = [
        {
            "dish": s.dish,
            "played": s.times_selected,
            "solved": s.times_solved,
            "solve_rate": round(s.solve_rate, 3),
            "avg_clues": round(s.total_clues / s.times_selected, 2) if s.times_selected else None,
        }
        for s in stats
    ]
    by_rate = sorted(scored, key=lambda r: (r["solve_rate"], -r["played"]))
    by_played = sorted(scored, key=lambda r: -r["played"])

    return AnalyticsOut(
        games_played=games_played,
        games_won=games_won,
        win_rate=_round(games_won / games_played) if games_played else 0.0,
        avg_clues_to_solve=_round(avg_clues, 2),
        avg_questions_to_read_mind=_round(avg_questions, 2),
        mind_reader_accuracy=_round(mind_won / mind_total) if mind_total else None,
        classifier_feedback_accuracy=(
            _round(feedback_right / feedback_total) if feedback_total else None
        ),
        hardest_dishes=by_rate[:8],
        easiest_dishes=list(reversed(by_rate))[:8],
        most_played=by_played[:8],
    )


@router.get("/predictions")
def recent_predictions(limit: int = 25, db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    ).all()
    return {
        "predictions": [
            {
                "id": r.id,
                "text": r.input_text,
                "top": r.top_label,
                "confidence": round(r.top_prob, 4),
                "latency_ms": round(r.latency_ms, 2),
                "at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.get("/feedback/export")
def export_feedback(db: Session = Depends(get_db)) -> dict:
    """Everything a retraining run needs, in corpus format."""
    rows = db.scalars(select(Feedback)).all()
    return {
        "rows": [
            {"text": r.input_text, "label": r.correct_label, "source": "feedback", "weight": 4.0}
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/answer-noise")
def answer_noise(min_answers: int = 5, db: Session = Depends(get_db)) -> dict:
    """How players actually answer, against what the catalogue says.

    The engine's accuracy target only holds while its answer-noise model is
    no more optimistic than real players. This reads every finished
    mind-reading round whose dish is known — the engine was confirmed right,
    or the player said what it was — and reports, per kind of question, how
    often a "yes" came back when the catalogue says yes and when it says no.
    Compare ``observed`` with ``assumed`` and move the settings in
    ``app/config.py`` if they have drifted apart.
    """
    catalogue = get_catalogue()
    table = question_table()

    sessions = db.scalars(
        select(GameSession).where(
            GameSession.mode == "mind",
            GameSession.status == "finished",
            GameSession.secret_dish.in_(catalogue.names),
        )
    ).all()

    def blank() -> dict:
        return {"yes_when_true": 0, "true": 0, "yes_when_false": 0, "false": 0, "unsure": 0}

    by_kind: dict[str, dict] = {}
    by_question: dict[str, dict] = {}
    for session in sessions:
        column = table.name_index[session.secret_dish]
        answers = db.scalars(
            select(Turn).where(Turn.session_id == session.id, Turn.kind == "answer")
        ).all()
        for turn in answers:
            key, answer = turn.payload.get("key"), turn.payload.get("answer")
            index = table.key_index.get(key)
            if index is None:
                continue
            truth = bool(table.truth[index, column])
            for bucket in (by_kind.setdefault(table.kinds[index], blank()),
                           by_question.setdefault(key, blank())):
                if answer == "unsure":
                    bucket["unsure"] += 1
                    continue
                bucket["true" if truth else "false"] += 1
                if answer == "yes":
                    bucket["yes_when_true" if truth else "yes_when_false"] += 1

    def rates(bucket: dict) -> dict:
        answered = bucket["true"] + bucket["false"]
        wrong = (bucket["true"] - bucket["yes_when_true"]) + bucket["yes_when_false"]
        return {
            "answers": answered,
            "unsure": bucket["unsure"],
            "p_yes_when_true": _round(bucket["yes_when_true"] / bucket["true"]) if bucket["true"] else None,
            "p_yes_when_false": _round(bucket["yes_when_false"] / bucket["false"]) if bucket["false"] else None,
            "wrong_rate": _round(wrong / answered) if answered else None,
        }

    assumed = {
        "attribute": [settings.p_correct, settings.p_flip],
        "origin": [settings.p_correct, settings.p_flip],
        "cuisine": [settings.cuisine_p_correct, settings.cuisine_p_flip],
        "ingredient": [settings.ingredient_p_correct, settings.ingredient_p_flip],
    }
    noisiest = sorted(
        (
            {"key": key, "question": catalogue.question_by_key[key].text, **rates(bucket)}
            for key, bucket in by_question.items()
            if bucket["true"] + bucket["false"] >= min_answers
        ),
        key=lambda row: -(row["wrong_rate"] or 0.0),
    )
    return {
        "rounds": len(sessions),
        "observed": {kind: rates(bucket) for kind, bucket in sorted(by_kind.items())},
        "assumed": {
            kind: {"p_yes_when_true": pair[0], "p_yes_when_false": pair[1]}
            for kind, pair in assumed.items()
        },
        "noisiest_questions": noisiest[:15],
    }
