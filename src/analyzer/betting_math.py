"""Expected value and Kelly stake sizing. Deterministic: no LLM arithmetic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.fetchers.odds_api import MatchOdds
from src.processing.markets import MARKET_LABELS, MARKET_ORDER


@dataclass(frozen=True)
class StakeSettings:
    bankroll: float = 1000.0
    kelly_multiplier: float = 0.25  # fractional Kelly; full Kelly is far too volatile for noisy estimates
    max_stake_fraction: float = 0.05  # hard cap per bet as a fraction of bankroll
    min_edge: float = 0.03  # minimum expected value per unit staked
    max_model_market_gap: float = 0.15  # larger gaps usually mean the model is wrong, not the market

    def __post_init__(self) -> None:
        if self.bankroll < 0:
            raise ValueError("bankroll must be >= 0")
        if not 0 <= self.kelly_multiplier <= 1:
            raise ValueError("kelly_multiplier must be in [0, 1]")
        if not 0 <= self.max_stake_fraction <= 1:
            raise ValueError("max_stake_fraction must be in [0, 1]")


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def expected_value(probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked: p * odds - 1."""
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll: (b*p - q) / b with b = odds - 1. Never negative."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max((b * probability - (1.0 - probability)) / b, 0.0)


@dataclass
class BetEvaluation:
    market: str
    label: str
    model_probability: float
    best_odds: float | None
    bookmaker: str | None
    implied_probability: float | None
    market_fair_probability: float | None
    expected_value: float | None
    kelly_full: float
    stake_fraction: float
    stake_amount: float
    is_value: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_markets(probabilities: dict[str, float], odds: MatchOdds, settings: StakeSettings) -> list[BetEvaluation]:
    """Evaluate every market against the best available price."""
    evaluations: list[BetEvaluation] = []
    for market in MARKET_ORDER:
        probability = probabilities.get(market)
        if probability is None:
            continue
        price = odds.best_prices.get(market)
        fair = odds.consensus_probabilities.get(market)
        if price is None:
            evaluations.append(
                BetEvaluation(market, MARKET_LABELS[market], probability, None, None, None, fair, None, 0.0, 0.0, 0.0, False, "no odds available")
            )
            continue

        ev = expected_value(probability, price.price)
        full_kelly = kelly_fraction(probability, price.price)
        gap = abs(probability - fair) if fair is not None else None

        if odds.is_mock:
            is_value, reason = False, "mock odds: illustrative only"
        elif ev < settings.min_edge:
            is_value, reason = False, f"EV below {settings.min_edge:.0%} threshold"
        elif gap is not None and gap > settings.max_model_market_gap:
            is_value, reason = False, f"model differs from market by {gap:.0%}: likely model error"
        else:
            is_value, reason = True, "value"

        stake_fraction = min(full_kelly * settings.kelly_multiplier, settings.max_stake_fraction) if is_value else 0.0
        evaluations.append(
            BetEvaluation(
                market=market,
                label=MARKET_LABELS[market],
                model_probability=probability,
                best_odds=price.price,
                bookmaker=price.bookmaker,
                implied_probability=implied_probability(price.price),
                market_fair_probability=fair,
                expected_value=ev,
                kelly_full=full_kelly,
                stake_fraction=stake_fraction,
                stake_amount=round(settings.bankroll * stake_fraction, 2),
                is_value=is_value,
                reason=reason,
            )
        )
    return evaluations
