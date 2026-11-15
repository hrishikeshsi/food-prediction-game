"""The fitted decision tree, for inspection."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.services.catalog import get_catalogue
from app.services.tree import canonical_tree, forest

router = APIRouter(prefix="/api/tree", tags=["tree"])


@router.get("")
def tree() -> dict:
    """The canonical scikit-learn tree as nested JSON, with its summary figures."""
    canonical = canonical_tree()
    return {
        "model": "sklearn.tree.DecisionTreeClassifier",
        "stats": canonical.stats(),
        "hint_trees": len(forest()),
        "tree": canonical.to_dict(),
    }


@router.get("/text", response_class=PlainTextResponse)
def tree_text() -> str:
    """The same tree as an indented outline."""
    return canonical_tree().to_text()


@router.get("/engine")
def engine() -> dict:
    """How the two engines are configured, and what the question bank holds."""
    catalogue = get_catalogue()
    kinds: dict[str, int] = {}
    for question in catalogue.questions:
        kinds[question.kind] = kinds.get(question.kind, 0) + 1
    return {
        "dishes": len(catalogue.dishes),
        "questions": len(catalogue.questions),
        "questions_by_kind": kinds,
        "forward": {
            "guess_threshold": settings.guess_threshold,
            "max_questions": settings.max_questions,
            "max_guesses": settings.max_guesses,
            "min_gain_bits": settings.min_gain_bits,
            "answer_model": {
                "attribute": [settings.p_correct, settings.p_flip],
                "origin": [settings.p_correct, settings.p_flip],
                "cuisine": [settings.cuisine_p_correct, settings.cuisine_p_flip],
                "ingredient": [settings.ingredient_p_correct, settings.ingredient_p_flip],
            },
        },
        "reverse": {
            "hint_trees": len(forest()),
            "ladder_floor": settings.ladder_floor,
        },
    }
