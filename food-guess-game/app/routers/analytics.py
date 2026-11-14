"""Aggregate statistics over everything the database has recorded."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import DishStat, Feedback, GameSession, Prediction
from app.schemas import AnalyticsOut

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
