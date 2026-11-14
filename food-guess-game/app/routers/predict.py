"""Direct access to the classifier: describe a dish, get ranked predictions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import Feedback, Prediction
from app.schemas import Candidate, FeedbackIn, PredictIn, PredictOut
from app.services.catalog import get_catalogue
from app.services.inference import get_predictor

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictOut)
def predict(body: PredictIn, db: Session = Depends(get_db)) -> PredictOut:
    predictor = get_predictor()
    results, latency = predictor.predict(body.text, top_k=body.top_k)
    if not results:
        raise HTTPException(400, "could not read anything usable from that description")

    row = Prediction(
        session_id=body.session_id,
        input_text=body.text,
        top_label=results[0].dish,
        top_prob=results[0].probability,
        topk=[{"dish": r.dish, "p": round(r.probability, 5)} for r in results],
        latency_ms=latency,
    )
    db.add(row)
    db.flush()

    return PredictOut(
        predictions=[
            Candidate(dish=r.dish, probability=round(r.probability, 5)) for r in results
        ],
        latency_ms=round(latency, 2),
        model_ready=predictor.ready,
        prediction_id=row.id,
    )


@router.post("/feedback")
def submit_feedback(body: FeedbackIn, db: Session = Depends(get_db)) -> dict:
    """Record what the dish actually was.

    This is the loop that makes the system improve: exported feedback rows can
    be appended to the corpus and folded into the next training run.
    """
    catalogue = get_catalogue()
    if body.correct_label not in catalogue.by_name:
        raise HTTPException(400, f"{body.correct_label!r} is not in the catalogue")

    row = Feedback(
        prediction_id=body.prediction_id,
        session_id=body.session_id,
        input_text=body.input_text,
        predicted_label=body.predicted_label,
        correct_label=body.correct_label,
        was_correct=body.predicted_label == body.correct_label,
    )
    db.add(row)
    db.flush()
    return {"ok": True, "feedback_id": row.id, "was_correct": row.was_correct}


@router.post("/reload")
def reload_model() -> dict:
    """Hot-swap freshly trained artifacts without restarting the server."""
    predictor = get_predictor()
    predictor.reload()
    return {"ready": predictor.ready, "note": predictor.reason}


@router.get("/dishes")
def list_dishes(q: str | None = None, limit: int = 50) -> dict:
    """Autocomplete source for the feedback and guess inputs."""
    catalogue = get_catalogue()
    names = catalogue.names
    if q:
        needle = q.lower()
        names = [n for n in names if needle in n.lower()]
    return {"dishes": sorted(names)[:limit], "total": len(catalogue.names)}
