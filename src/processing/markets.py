"""Betting market identifiers shared by the model, the odds fetcher and the analyzer."""
from __future__ import annotations

TOTALS_LINES: tuple[float, ...] = (1.5, 2.5, 3.5)


def totals_market_id(side: str, line: float) -> str:
    """totals_market_id("over", 2.5) -> "over_2_5"."""
    return f"{side}_{str(line).replace('.', '_')}"


MARKET_ORDER: tuple[str, ...] = (
    "home",
    "draw",
    "away",
    *(totals_market_id(side, line) for line in TOTALS_LINES for side in ("over", "under")),
    "btts_yes",
    "btts_no",
)

MARKET_LABELS: dict[str, str] = {
    "home": "Home win",
    "draw": "Draw",
    "away": "Away win",
    **{
        totals_market_id(side, line): f"{side.capitalize()} {line} goals"
        for line in TOTALS_LINES
        for side in ("over", "under")
    },
    "btts_yes": "BTTS - Yes",
    "btts_no": "BTTS - No",
}

# Mutually exclusive, exhaustive outcome sets; used to strip the bookmaker margin.
MARKET_GROUPS: tuple[tuple[str, ...], ...] = (
    ("home", "draw", "away"),
    *((totals_market_id("over", line), totals_market_id("under", line)) for line in TOTALS_LINES),
    ("btts_yes", "btts_no"),
)
