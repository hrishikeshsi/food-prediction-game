"""Chef Mystico — FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
or:
    ./run.sh
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import analytics, game, predict, tree
from app.schemas import HealthOut
from app.services.catalog import get_catalogue
from app.services.engine import question_table
from app.services.inference import get_predictor
from app.services.tree import canonical_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("chef-mystico")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    catalogue = get_catalogue()
    question_table()  # warm the precomputed question/dish truth table
    stats = canonical_tree().stats()  # fit (or load) the decision trees
    predictor = get_predictor()
    log.info(
        "ready — %d dishes, %d questions, tree depth %.2f mean / %d max, model_ready=%s",
        len(catalogue.dishes),
        len(catalogue.questions),
        stats["mean_depth"],
        stats["max_depth"],
        predictor.ready,
    )
    if not predictor.ready:
        log.warning("running without a trained model: %s", predictor.reason)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "A bidirectional food guessing game. Forward: a Bayesian decision engine "
        "names the dish you are thinking of. Reverse: it hides a dish and reveals "
        "hints along a fitted decision tree. Plus a direct interface to the "
        "TF-IDF + PyTorch dish classifier."
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
app.include_router(tree.router)


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


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
# The game used to be two pages in ``food-prediction-game/data``. Both now open
# the single-page app in ``frontend/``, in the matching mode.
OLD_GAME_PAGES = {"chef-guess.html": "forward", "you-guess.html": "reverse"}


@app.get("/data/{page}", include_in_schema=False)
def old_game_page(page: str) -> RedirectResponse:
    if page not in OLD_GAME_PAGES:
        raise HTTPException(404, "no such page")
    return RedirectResponse(f"/#{OLD_GAME_PAGES[page]}")


@app.get("/tree", include_in_schema=False)
def tree_page() -> FileResponse:
    return FileResponse(settings.frontend_dir / "tree.html")


if settings.frontend_dir.exists():
    # Everything else is the front end, served at the same relative paths it
    # has on disk — so the pages work the same here, from Live Server, or
    # opened straight from the folder. Mounted last, after every API route.
    app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
else:  # pragma: no cover

    @app.get("/", include_in_schema=False)
    def index() -> JSONResponse:
        return JSONResponse({"message": f"{settings.app_name} API", "docs": "/docs"})
