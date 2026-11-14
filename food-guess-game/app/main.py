"""Chef Mystico — FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
or:
    ./run.sh
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import analytics, game, predict
from app.schemas import HealthOut
from app.services.catalog import get_catalogue
from app.services.engine import predicate_matrix
from app.services.inference import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("chef-mystico")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    catalogue = get_catalogue()
    predicate_matrix()  # warm the precomputed question/dish truth table
    predictor = get_predictor()
    log.info(
        "ready — %d dishes, %d questions, model_ready=%s",
        len(catalogue.dishes),
        len(catalogue.questions),
        predictor.ready,
    )
    if not predictor.ready:
        log.warning("running without a trained model: %s", predictor.reason)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "A hard-mode food guessing game. Two play modes plus a direct interface "
        "to the TF-IDF + PyTorch dish classifier."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game.router)
app.include_router(predict.router)
app.include_router(analytics.router)


@app.get("/api/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    catalogue = get_catalogue()
    predictor = get_predictor()
    metrics = predictor.metrics or None
    if metrics:
        metrics = {
            k: metrics[k]
            for k in ("val_top1", "val_top3", "val_top5", "val_macro_f1",
                      "labels", "features", "parameters", "trained_at")
            if k in metrics
        }
    return HealthOut(
        status="ok",
        version=settings.version,
        dishes=len(catalogue.dishes),
        questions=len(catalogue.questions),
        model_ready=predictor.ready,
        model_note=predictor.reason,
        model_metrics=metrics,
    )


if settings.frontend_dir.exists():
    app.mount(
        "/static", StaticFiles(directory=str(settings.frontend_dir)), name="static"
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.frontend_dir / "index.html")
else:  # pragma: no cover

    @app.get("/", include_in_schema=False)
    def index() -> JSONResponse:
        return JSONResponse({"message": f"{settings.app_name} API", "docs": "/docs"})
