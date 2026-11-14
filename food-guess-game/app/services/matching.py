"""Fuzzy matching of a typed guess against the catalogue.

Players will type "biriyani", "cha wan mushi", "pierogies" or just "that Polish
cabbage stew". Exact string equality would reject all of those, so guesses go
through a cascade: exact surface form, then token containment, then a character
similarity ratio, then a token-overlap fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.services.catalog import Catalogue, normalise

# Words that carry no identifying power in a guess.
STOPWORDS = {
    "the", "a", "an", "of", "with", "and", "is", "it", "that", "this", "some",
    "dish", "food", "style", "maybe", "probably", "think", "its", "i",
}

ACCEPT_THRESHOLD = 0.82


@dataclass
class MatchResult:
    dish: str | None
    score: float
    method: str

    @property
    def accepted(self) -> bool:
        return self.dish is not None and self.score >= ACCEPT_THRESHOLD


def _tokens(text: str) -> set[str]:
    return {t for t in normalise(text).split() if t not in STOPWORDS and len(t) > 2}


def match_guess(guess: str, catalogue: Catalogue, restrict_to: str | None = None) -> MatchResult:
    """Resolve a free-text guess to a canonical dish name."""
    probe = normalise(guess)
    if not probe:
        return MatchResult(None, 0.0, "empty")

    surfaces = catalogue.surface_index
    if restrict_to:
        target = catalogue.get(restrict_to)
        surfaces = {
            normalise(s): restrict_to
            for s in [target["name"], *target.get("aliases", [])]
        }

    # 1. Exact surface form.
    if probe in surfaces:
        return MatchResult(surfaces[probe], 1.0, "exact")

    # 2. Whole surface form appearing inside a longer sentence, or vice versa.
    best_containment: tuple[str, float] | None = None
    for surface, name in surfaces.items():
        if len(surface) < 4:
            continue
        if surface in probe or (len(probe) >= 4 and probe in surface):
            ratio = min(len(surface), len(probe)) / max(len(surface), len(probe))
            score = 0.90 + 0.10 * ratio
            if best_containment is None or score > best_containment[1]:
                best_containment = (name, score)
    if best_containment:
        return MatchResult(best_containment[0], round(best_containment[1], 4), "containment")

    # 3. Character-level similarity, for misspellings and transliteration drift.
    best_name, best_score = None, 0.0
    for surface, name in surfaces.items():
        score = SequenceMatcher(None, probe, surface).ratio()
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= ACCEPT_THRESHOLD:
        return MatchResult(best_name, round(best_score, 4), "fuzzy")

    # 4. Token overlap, for descriptive guesses like "polish cabbage stew".
    probe_tokens = _tokens(guess)
    if probe_tokens:
        overlap_name, overlap_score = None, 0.0
        for surface, name in surfaces.items():
            surface_tokens = _tokens(surface)
            if not surface_tokens:
                continue
            shared = probe_tokens & surface_tokens
            if not shared:
                continue
            score = len(shared) / len(surface_tokens | probe_tokens)
            if score > overlap_score:
                overlap_name, overlap_score = name, score
        if overlap_score >= 0.5:
            return MatchResult(overlap_name, round(0.80 + 0.15 * overlap_score, 4), "tokens")

    # Nothing convincing — return the nearest miss so the UI can say
    # "did you mean ...?" without accepting it.
    return MatchResult(best_name, round(best_score, 4), "near-miss")
