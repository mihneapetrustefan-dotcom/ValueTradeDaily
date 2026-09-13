"""Independent-Poisson goal model.

Each team's goals are modelled as Poisson(xG). The joint score distribution is the
outer product of the two marginal distributions; every market probability is a sum
over the relevant cells of that matrix.

The matrix is computed up to ``max_goals`` (default 10) so that almost no probability
mass is lost, then renormalised. The 0-5 x 0-5 block is returned for display.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import poisson

from src.processing.markets import TOTALS_LINES, totals_market_id

DEFAULT_MAX_GOALS = 10
DISPLAY_MAX_GOALS = 5


def _validate_xg(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return value


def score_matrix(home_xg: float, away_xg: float, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Joint probability matrix P[h, a] for h, a in 0..max_goals (rows = home goals)."""
    home_xg = _validate_xg(home_xg, "home_xg")
    away_xg = _validate_xg(away_xg, "away_xg")
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, home_xg)
    away_pmf = poisson.pmf(goals, away_xg)
    return np.outer(home_pmf, away_pmf)


def predict(
    home_xg: float,
    away_xg: float,
    max_goals: int = DEFAULT_MAX_GOALS,
    display_max_goals: int = DISPLAY_MAX_GOALS,
) -> dict[str, Any]:
    """Market probabilities from home and away expected goals.

    Returns a dict with:
      - ``probabilities``: raw probabilities in [0, 1] keyed by market id
      - ``percentages``: the same values as percentages rounded to 2 decimals
      - ``fair_odds``: 1 / probability
      - ``score_matrix``: 0..display_max_goals grid of exact-score percentages
      - ``most_likely_scores``: top five exact scores
    """
    if max_goals < display_max_goals:
        raise ValueError("max_goals must be >= display_max_goals")

    matrix = score_matrix(home_xg, away_xg, max_goals)
    captured_mass = float(matrix.sum())
    matrix = matrix / captured_mass

    goals = np.arange(max_goals + 1)
    total_goals = goals[:, None] + goals[None, :]

    probabilities: dict[str, float] = {
        "home": float(np.tril(matrix, k=-1).sum()),  # home goals > away goals
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, k=1).sum()),  # away goals > home goals
    }
    for line in TOTALS_LINES:
        over = float(matrix[total_goals > line].sum())
        probabilities[totals_market_id("over", line)] = over
        probabilities[totals_market_id("under", line)] = 1.0 - over

    btts_yes = float(matrix[1:, 1:].sum())
    probabilities["btts_yes"] = btts_yes
    probabilities["btts_no"] = 1.0 - btts_yes

    display = matrix[: display_max_goals + 1, : display_max_goals + 1] * 100.0

    flat_order = np.argsort(matrix, axis=None)[::-1][:5]
    most_likely = [
        {"score": f"{h}-{a}", "percentage": round(float(matrix[h, a]) * 100.0, 2)}
        for h, a in (np.unravel_index(idx, matrix.shape) for idx in flat_order)
    ]

    return {
        "inputs": {"home_xg": float(home_xg), "away_xg": float(away_xg), "max_goals": max_goals},
        "probabilities": probabilities,
        "percentages": {market: round(prob * 100.0, 2) for market, prob in probabilities.items()},
        "fair_odds": {market: (round(1.0 / prob, 3) if prob > 0 else None) for market, prob in probabilities.items()},
        "score_matrix": {
            "rows": "home goals",
            "columns": "away goals",
            "max_goals": display_max_goals,
            "percentages": [[round(float(cell), 2) for cell in row] for row in display],
        },
        "most_likely_scores": most_likely,
        "truncated_mass_pct": round((1.0 - captured_mass) * 100.0, 6),
    }
