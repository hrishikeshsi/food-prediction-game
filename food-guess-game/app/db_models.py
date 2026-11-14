"""ORM models.

Everything a game does is written down: the secret dish, every clue revealed,
every question the engine asked and how it was answered, every guess, and every
model prediction with its confidence. That history is what the analytics
endpoints read, and what a future retraining run can learn from.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GameSession(Base):
    __tablename__ = "game_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    client_id: Mapped[str | None] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(24), index=True)

    secret_dish: Mapped[str | None] = mapped_column(String(120), index=True)
    difficulty_min: Mapped[int] = mapped_column(Integer, default=1)
    difficulty_max: Mapped[int] = mapped_column(Integer, default=5)
    region: Mapped[str | None] = mapped_column(String(48))

    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    outcome: Mapped[str | None] = mapped_column(String(24))
    score: Mapped[int] = mapped_column(Integer, default=0)

    clues_revealed: Mapped[int] = mapped_column(Integer, default=0)
    questions_asked: Mapped[int] = mapped_column(Integer, default=0)
    wrong_guesses: Mapped[int] = mapped_column(Integer, default=0)

    # Live Bayesian belief for the mind-reading mode, persisted so the game
    # survives a server restart mid-round.
    belief: Mapped[dict | None] = mapped_column(JSON)
    asked_keys: Mapped[list | None] = mapped_column(JSON, default=list)
    rejected: Mapped[list | None] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    turns: Mapped[list["Turn"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Turn.index"
    )
    guesses: Mapped[list["Guess"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.created_at).total_seconds()


class Turn(Base):
    """One atomic event in a game: a clue, a question, an answer, a reveal."""

    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("game_sessions.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(24), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[GameSession] = relationship(back_populates="turns")


class Guess(Base):
    __tablename__ = "guesses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("game_sessions.id", ondelete="CASCADE"), index=True
    )
    raw_text: Mapped[str] = mapped_column(Text)
    matched_dish: Mapped[str | None] = mapped_column(String(120), index=True)
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    correct: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[GameSession] = relationship(back_populates="guesses")


class Prediction(Base):
    """A call to the classifier, kept so accuracy can be tracked over time."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(32), index=True)
    input_text: Mapped[str] = mapped_column(Text)
    top_label: Mapped[str] = mapped_column(String(120), index=True)
    top_prob: Mapped[float] = mapped_column(Float)
    topk: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Feedback(Base):
    """Player-supplied ground truth for a prediction — the retraining signal."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("predictions.id", ondelete="SET NULL"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(32), index=True)
    input_text: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[str | None] = mapped_column(String(120))
    correct_label: Mapped[str] = mapped_column(String(120), index=True)
    was_correct: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DishStat(Base):
    """Rolling per-dish aggregates, updated as games finish."""

    __tablename__ = "dish_stats"

    dish: Mapped[str] = mapped_column(String(120), primary_key=True)
    times_selected: Mapped[int] = mapped_column(Integer, default=0)
    times_solved: Mapped[int] = mapped_column(Integer, default=0)
    total_clues: Mapped[int] = mapped_column(Integer, default=0)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    total_wrong_guesses: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    @property
    def solve_rate(self) -> float:
        return self.times_solved / self.times_selected if self.times_selected else 0.0


Index("ix_turns_session_index", Turn.session_id, Turn.index)
Index("ix_sessions_mode_status", GameSession.mode, GameSession.status)
