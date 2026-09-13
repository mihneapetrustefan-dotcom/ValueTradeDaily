"""Bookmaker odds from The Odds API (https://the-odds-api.com).

Fetches 1X2 (h2h), Over/Under (totals) and, optionally, BTTS prices for a fixture,
keeps the best price per outcome across bookmakers, and derives margin-free
consensus probabilities. When no API key is configured, the request fails, or the
fixture is not listed, a mock set of prices is returned instead. Mock odds are
always flagged with ``is_mock=True`` and must never be used to size real stakes.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from src.config import LEAGUES, League, get_env, resolve_league
from src.processing.markets import MARKET_GROUPS, TOTALS_LINES, totals_market_id
from src.processing.team_names import MATCH_THRESHOLD, team_similarity

logger = logging.getLogger(__name__)

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "eu"
MOCK_BOOKMAKER = "Mock prices (not real)"

# Generic prices shaped like a typical top-flight fixture. Illustrative only.
MOCK_PRICES: dict[str, float] = {
    "home": 2.40,
    "draw": 3.40,
    "away": 3.00,
    "over_1_5": 1.30,
    "under_1_5": 3.50,
    "over_2_5": 1.90,
    "under_2_5": 1.95,
    "over_3_5": 3.20,
    "under_3_5": 1.35,
    "btts_yes": 1.80,
    "btts_no": 1.95,
}


class OddsAPIError(RuntimeError):
    """Raised when The Odds API cannot be reached or returns an error."""


@dataclass
class OutcomePrice:
    price: float
    bookmaker: str


@dataclass
class MatchOdds:
    home_team: str
    away_team: str
    source: str
    is_mock: bool
    sport_key: str | None = None
    event_id: str | None = None
    commence_time: str | None = None
    best_prices: dict[str, OutcomePrice] = field(default_factory=dict)
    consensus_probabilities: dict[str, float] = field(default_factory=dict)
    bookmaker_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    requests_remaining: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def bookmaker_count(self) -> int:
        return len(self.bookmaker_prices)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bookmaker_count"] = self.bookmaker_count
        return data


def best_prices(bookmaker_prices: dict[str, dict[str, float]], markets: tuple[str, ...] | None = None) -> dict[str, OutcomePrice]:
    """Highest decimal price per market across all bookmakers."""
    best: dict[str, OutcomePrice] = {}
    for bookmaker, prices in bookmaker_prices.items():
        for market, price in prices.items():
            if markets is not None and market not in markets:
                continue
            if market not in best or price > best[market].price:
                best[market] = OutcomePrice(price=price, bookmaker=bookmaker)
    return best


def consensus_probabilities(bookmaker_prices: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average margin-free probability per outcome.

    For each bookmaker quoting a complete outcome group (e.g. home/draw/away), the
    implied probabilities are rescaled to sum to 1 (proportional de-vig), then
    averaged across bookmakers.
    """
    result: dict[str, float] = {}
    for group in MARKET_GROUPS:
        samples: list[list[float]] = []
        for prices in bookmaker_prices.values():
            if all(market in prices for market in group):
                implied = [1.0 / prices[market] for market in group]
                overround = sum(implied)
                samples.append([p / overround for p in implied])
        if not samples:
            continue
        averaged = [sum(sample[i] for sample in samples) / len(samples) for i in range(len(group))]
        total = sum(averaged)
        for market, prob in zip(group, averaged):
            result[market] = prob / total
    return result


def _as_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 1.0 else None


def mock_odds(home_team: str, away_team: str, reason: str) -> MatchOdds:
    """Clearly-flagged placeholder odds used when real prices are unavailable."""
    prices = {MOCK_BOOKMAKER: dict(MOCK_PRICES)}
    return MatchOdds(
        home_team=home_team,
        away_team=away_team,
        source="mock",
        is_mock=True,
        best_prices=best_prices(prices),
        consensus_probabilities=consensus_probabilities(prices),
        bookmaker_prices=prices,
        notes=[reason, "Mock odds are illustrative only: no stake will be recommended."],
    )


class OddsAPIClient:
    """Thin, defensive client for The Odds API v4."""

    def __init__(
        self,
        api_key: str | None = None,
        regions: str = DEFAULT_REGIONS,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else get_env("ODDS_API_KEY")
        self.regions = regions
        self.timeout = timeout
        self.session = session or requests.Session()
        self.requests_remaining: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not configured")
        url = f"{ODDS_API_BASE_URL}{path}"
        query = {"apiKey": self.api_key, **(params or {})}
        try:
            response = self.session.get(url, params=query, timeout=self.timeout)
        except requests.Timeout as exc:
            raise OddsAPIError(f"The Odds API timed out after {self.timeout:.0f}s") from exc
        except requests.RequestException as exc:
            raise OddsAPIError(f"network error: {self._redact(str(exc))}") from exc

        self.requests_remaining = response.headers.get("x-requests-remaining", self.requests_remaining)

        if response.status_code == 401:
            raise OddsAPIError("API key rejected (HTTP 401)")
        if response.status_code == 429:
            raise OddsAPIError("quota or rate limit exceeded (HTTP 429)")
        if not response.ok:
            raise OddsAPIError(f"HTTP {response.status_code}: {self._redact(_error_message(response))}")
        try:
            return response.json()
        except ValueError as exc:
            raise OddsAPIError("invalid JSON in response") from exc

    def get_events_with_odds(self, sport_key: str, markets: tuple[str, ...] = ("h2h", "totals")) -> list[dict[str, Any]]:
        """Upcoming events with odds for one competition. Costs len(markets) x regions credits."""
        data = self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions": self.regions,
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(data, list):
            raise OddsAPIError(f"unexpected response shape for {sport_key}")
        return data

    def get_event_odds(self, sport_key: str, event_id: str, markets: tuple[str, ...]) -> dict[str, Any]:
        """Odds for a single event; required for non-featured markets such as BTTS."""
        data = self._get(
            f"/sports/{sport_key}/events/{event_id}/odds",
            {
                "regions": self.regions,
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(data, dict):
            raise OddsAPIError(f"unexpected response shape for event {event_id}")
        return data

    @staticmethod
    def find_event(events: list[dict[str, Any]], home_team: str, away_team: str) -> tuple[dict[str, Any], bool] | None:
        """Locate a fixture by fuzzy team names. Returns (event, swapped) or None.

        ``swapped`` is True when the bookmaker lists the teams the other way round.
        """
        best: tuple[dict[str, Any], bool, float] | None = None
        for event in events:
            ev_home, ev_away = event.get("home_team", ""), event.get("away_team", "")
            straight = min(team_similarity(home_team, ev_home), team_similarity(away_team, ev_away))
            swapped = min(team_similarity(home_team, ev_away), team_similarity(away_team, ev_home))
            score, is_swapped = (straight, False) if straight >= swapped else (swapped, True)
            if best is None or score > best[2]:
                best = (event, is_swapped, score)
        if best is None or best[2] < MATCH_THRESHOLD:
            return None
        return best[0], best[1]

    @staticmethod
    def extract_bookmaker_prices(event: dict[str, Any], swapped: bool = False) -> dict[str, dict[str, float]]:
        """Map every bookmaker's h2h / totals / btts outcomes onto our market ids.

        Home/away ids always refer to the teams as the user entered them.
        """
        ev_home, ev_away = event.get("home_team"), event.get("away_team")
        result: dict[str, dict[str, float]] = {}
        for bookmaker in event.get("bookmakers", []) or []:
            title = bookmaker.get("title") or bookmaker.get("key") or "unknown"
            prices: dict[str, float] = {}
            for market in bookmaker.get("markets", []) or []:
                key = market.get("key")
                for outcome in market.get("outcomes", []) or []:
                    price = _as_price(outcome.get("price"))
                    if price is None:
                        continue
                    name = outcome.get("name")
                    market_id: str | None = None
                    if key == "h2h":
                        if name == "Draw":
                            market_id = "draw"
                        elif name == ev_home:
                            market_id = "away" if swapped else "home"
                        elif name == ev_away:
                            market_id = "home" if swapped else "away"
                    elif key == "totals":
                        point = outcome.get("point")
                        if name in ("Over", "Under") and isinstance(point, (int, float)) and float(point) in TOTALS_LINES:
                            market_id = totals_market_id(name.lower(), float(point))
                    elif key == "btts" and name in ("Yes", "No"):
                        market_id = f"btts_{name.lower()}"
                    if market_id:
                        prices[market_id] = price
            if prices:
                result[title] = prices
        return result

    def extract_1x2(self, event: dict[str, Any], swapped: bool = False) -> dict[str, OutcomePrice]:
        """Best home/draw/away prices for an event."""
        return best_prices(self.extract_bookmaker_prices(event, swapped), ("home", "draw", "away"))

    def extract_over_under(self, event: dict[str, Any], swapped: bool = False) -> dict[str, OutcomePrice]:
        """Best Over/Under prices for the 1.5, 2.5 and 3.5 lines."""
        markets = tuple(totals_market_id(side, line) for line in TOTALS_LINES for side in ("over", "under"))
        return best_prices(self.extract_bookmaker_prices(event, swapped), markets)

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        league: str | League | None = None,
        include_btts: bool = True,
    ) -> MatchOdds:
        """Best available odds for a fixture, falling back to flagged mock odds."""
        if not self.available:
            return mock_odds(home_team, away_team, "ODDS_API_KEY is not set, so real odds were not fetched.")

        if league is None:
            leagues = list(LEAGUES.values())
        else:
            leagues = [league if isinstance(league, League) else resolve_league(league)]

        found: tuple[dict[str, Any], bool] | None = None
        found_league: League | None = None
        try:
            for candidate in leagues:
                events = self.get_events_with_odds(candidate.odds_api_sport_key)
                found = self.find_event(events, home_team, away_team)
                if found:
                    found_league = candidate
                    break
        except OddsAPIError as exc:
            logger.warning("Odds API request failed: %s", exc)
            return mock_odds(home_team, away_team, f"The Odds API is unavailable ({exc}).")

        if not found or found_league is None:
            searched = ", ".join(lg.name for lg in leagues)
            return mock_odds(home_team, away_team, f"No upcoming fixture {home_team} vs {away_team} listed in: {searched}.")

        event, swapped = found
        notes: list[str] = []
        if swapped:
            notes.append(
                f"Bookmakers list this fixture as {event.get('home_team')} (home) vs {event.get('away_team')}; "
                "prices were mapped to the teams as you entered them, but check which side is really at home."
            )
        prices = self.extract_bookmaker_prices(event, swapped)

        if include_btts and event.get("id"):
            try:
                btts_event = self.get_event_odds(found_league.odds_api_sport_key, event["id"], ("btts",))
                for bookmaker, btts_prices in self.extract_bookmaker_prices(btts_event, swapped).items():
                    prices.setdefault(bookmaker, {}).update(btts_prices)
            except OddsAPIError as exc:
                notes.append(f"BTTS odds unavailable: {exc}.")

        if not prices:
            return mock_odds(home_team, away_team, "The fixture is listed but no bookmaker has priced it yet.")

        return MatchOdds(
            home_team=home_team,
            away_team=away_team,
            source="the-odds-api",
            is_mock=False,
            sport_key=found_league.odds_api_sport_key,
            event_id=event.get("id"),
            commence_time=event.get("commence_time"),
            best_prices=best_prices(prices),
            consensus_probabilities=consensus_probabilities(prices),
            bookmaker_prices=prices,
            requests_remaining=self.requests_remaining,
            notes=notes,
        )


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("message") or payload)[:200]
    return str(payload)[:200]
