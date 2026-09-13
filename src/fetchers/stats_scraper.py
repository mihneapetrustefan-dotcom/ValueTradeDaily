"""Team statistics and team news.

``get_match_stats`` aggregates what can be obtained legitimately and for free:
league tables with home/away splits and recent results from the football-data.org
v4 API. From those it derives a goals-based expected-goals estimate (attack and
defence strength relative to the league average, shrunk towards average while the
sample is small). Shot-based xG, confirmed lineups and card statistics are not
available from free sources; they are reported as unavailable rather than invented.

``get_team_news`` pulls the latest headlines for a team from DuckDuckGo News.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.config import CACHE_DIR, LEAGUES, get_env, resolve_league
from src.processing.team_names import MATCH_THRESHOLD, team_similarity

logger = logging.getLogger(__name__)

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
STANDINGS_CACHE_SECONDS = 6 * 3600
MATCHES_CACHE_SECONDS = 3600
# Pseudo-games of league-average performance blended into each team's rates.
SHRINKAGE_GAMES = 6.0
XG_FLOOR, XG_CEILING = 0.2, 4.0
RECENT_MATCHES = 5


class FootballDataError(RuntimeError):
    """Raised when football-data.org cannot be reached or returns an error."""


@dataclass
class SplitRecord:
    played: int
    goals_for: int
    goals_against: int

    @property
    def goals_for_per_game(self) -> float | None:
        return self.goals_for / self.played if self.played else None

    @property
    def goals_against_per_game(self) -> float | None:
        return self.goals_against / self.played if self.played else None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "goals_for_per_game": _round(self.goals_for_per_game),
            "goals_against_per_game": _round(self.goals_against_per_game),
        }


@dataclass
class TeamStanding:
    team_id: int
    name: str
    short_name: str | None
    tla: str | None
    competition_code: str
    competition_name: str
    position: int | None
    points: int | None
    won: int | None
    draw: int | None
    lost: int | None
    form: str | None
    total: SplitRecord
    home: SplitRecord | None = None
    away: SplitRecord | None = None


@dataclass
class LeagueTable:
    code: str
    name: str
    season_start: str | None
    teams: dict[int, TeamStanding]
    avg_home_goals: float | None
    avg_away_goals: float | None


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _retry_after_seconds(response: requests.Response) -> float:
    header = response.headers.get("X-RequestCounter-Reset")
    if header and header.strip().isdigit():
        return min(max(float(header), 1.0), 65.0)
    match = re.search(r"(\d+)\s*seconds?", response.text or "")
    return min(max(float(match.group(1)), 1.0), 65.0) if match else 60.0


class FootballDataClient:
    """Client for football-data.org v4 with an on-disk JSON cache and 429 back-off."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        cache_dir: Path = CACHE_DIR,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else get_env("FOOTBALL_DATA_API_KEY")
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.session = session or requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        key = hashlib.sha1(f"{path}?{sorted((params or {}).items())}".encode()).hexdigest()
        return self.cache_dir / f"football_data_{key}.json"

    def _get(self, path: str, params: dict[str, Any] | None = None, cache_seconds: float = 0) -> dict[str, Any]:
        cache_path = self._cache_path(path, params) if cache_seconds > 0 else None
        if cache_path and cache_path.exists() and time.time() - cache_path.stat().st_mtime < cache_seconds:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.debug("Ignoring unreadable cache file %s", cache_path)

        if not self.api_key:
            raise FootballDataError("FOOTBALL_DATA_API_KEY is not configured")

        url = f"{FOOTBALL_DATA_BASE_URL}{path}"
        response: requests.Response | None = None
        for attempt in range(2):
            try:
                response = self.session.get(
                    url, params=params, headers={"X-Auth-Token": self.api_key}, timeout=self.timeout
                )
            except requests.Timeout as exc:
                raise FootballDataError(f"football-data.org timed out after {self.timeout:.0f}s") from exc
            except requests.RequestException as exc:
                raise FootballDataError(f"network error: {exc}") from exc
            if response.status_code == 429 and attempt == 0:
                wait = _retry_after_seconds(response)
                logger.warning("football-data.org rate limit hit; waiting %.0fs", wait)
                time.sleep(wait)
                continue
            break

        assert response is not None
        if response.status_code in (401, 403):
            raise FootballDataError(f"access denied for {path} (HTTP {response.status_code}): {_message(response)}")
        if response.status_code == 404:
            raise FootballDataError(f"not found: {path}")
        if response.status_code == 429:
            raise FootballDataError("rate limit exceeded (HTTP 429)")
        if not response.ok:
            raise FootballDataError(f"HTTP {response.status_code} for {path}: {_message(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise FootballDataError(f"invalid JSON from {path}") from exc

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            except OSError:
                logger.debug("Could not write cache file %s", cache_path)
        return data

    def get_league_table(self, competition_code: str) -> LeagueTable:
        data = self._get(f"/competitions/{competition_code}/standings", cache_seconds=STANDINGS_CACHE_SECONDS)
        return parse_standings(data, competition_code)

    def get_team_matches(self, team_id: int, date_from: date, date_to: date) -> list[dict[str, Any]]:
        data = self._get(
            f"/teams/{team_id}/matches",
            {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
            cache_seconds=MATCHES_CACHE_SECONDS,
        )
        return list(data.get("matches", []))


def _message(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("message", payload))[:200] if isinstance(payload, dict) else str(payload)[:200]
    except ValueError:
        return (response.text or "")[:200]


def _split(row: dict[str, Any]) -> SplitRecord:
    return SplitRecord(
        played=int(row.get("playedGames") or 0),
        goals_for=int(row.get("goalsFor") or 0),
        goals_against=int(row.get("goalsAgainst") or 0),
    )


def parse_standings(data: dict[str, Any], competition_code: str) -> LeagueTable:
    """Convert a /standings payload into a LeagueTable with home/away splits."""
    competition = data.get("competition") or {}
    teams: dict[int, TeamStanding] = {}
    home_rows: dict[int, dict[str, Any]] = {}
    away_rows: dict[int, dict[str, Any]] = {}

    for standing in data.get("standings", []) or []:
        kind = standing.get("type")
        for row in standing.get("table", []) or []:
            team = row.get("team") or {}
            team_id = team.get("id")
            if team_id is None:
                continue
            if kind == "HOME":
                home_rows[team_id] = row
            elif kind == "AWAY":
                away_rows[team_id] = row
            elif kind == "TOTAL" and team_id not in teams:
                teams[team_id] = TeamStanding(
                    team_id=team_id,
                    name=team.get("name") or "",
                    short_name=team.get("shortName"),
                    tla=team.get("tla"),
                    competition_code=competition_code,
                    competition_name=competition.get("name") or competition_code,
                    position=row.get("position"),
                    points=row.get("points"),
                    won=row.get("won"),
                    draw=row.get("draw"),
                    lost=row.get("lost"),
                    form=row.get("form"),
                    total=_split(row),
                )

    for team_id, standing in teams.items():
        if team_id in home_rows:
            standing.home = _split(home_rows[team_id])
        if team_id in away_rows:
            standing.away = _split(away_rows[team_id])

    home_played = sum(int(r.get("playedGames") or 0) for r in home_rows.values())
    away_played = sum(int(r.get("playedGames") or 0) for r in away_rows.values())
    avg_home = sum(int(r.get("goalsFor") or 0) for r in home_rows.values()) / home_played if home_played else None
    avg_away = sum(int(r.get("goalsFor") or 0) for r in away_rows.values()) / away_played if away_played else None

    return LeagueTable(
        code=competition_code,
        name=competition.get("name") or competition_code,
        season_start=(data.get("season") or {}).get("startDate"),
        teams=teams,
        avg_home_goals=avg_home,
        avg_away_goals=avg_away,
    )


def find_team(table: LeagueTable, query: str) -> TeamStanding | None:
    best: tuple[TeamStanding, float] | None = None
    for standing in table.teams.values():
        if standing.tla and query.strip().upper() == standing.tla:
            return standing
        score = max(team_similarity(query, n) for n in (standing.name, standing.short_name or "") if n) if standing.name else 0.0
        if best is None or score > best[1]:
            best = (standing, score)
    return best[0] if best and best[1] >= MATCH_THRESHOLD else None


def _competition_search_order(league: str | None) -> list[str]:
    """Domestic leagues first (better strength data), the requested one leading, UCL last."""
    domestic = [lg.football_data_code for lg in LEAGUES.values() if lg.domestic]
    order = list(domestic)
    if league:
        requested = resolve_league(league)
        if requested.domestic:
            order.remove(requested.football_data_code)
            order.insert(0, requested.football_data_code)
    order.append(LEAGUES["ucl"].football_data_code)
    return order


def _shrunk_ratio(team_rate: float | None, league_rate: float | None, games: int) -> float:
    if team_rate is None or not league_rate:
        return 1.0
    weight = games / (games + SHRINKAGE_GAMES)
    return weight * (team_rate / league_rate) + (1.0 - weight)


def estimate_expected_goals(
    home: TeamStanding, home_table: LeagueTable, away: TeamStanding, away_table: LeagueTable
) -> dict[str, Any] | None:
    """Goals-based xG estimate from home/away attack and defence strengths."""
    if not (home_table.avg_home_goals and home_table.avg_away_goals and away_table.avg_home_goals and away_table.avg_away_goals):
        return None
    home_split = home.home or home.total
    away_split = away.away or away.total

    league_home_goals = (home_table.avg_home_goals + away_table.avg_home_goals) / 2
    league_away_goals = (home_table.avg_away_goals + away_table.avg_away_goals) / 2

    home_attack = _shrunk_ratio(home_split.goals_for_per_game, home_table.avg_home_goals, home_split.played)
    away_defence = _shrunk_ratio(away_split.goals_against_per_game, away_table.avg_home_goals, away_split.played)
    away_attack = _shrunk_ratio(away_split.goals_for_per_game, away_table.avg_away_goals, away_split.played)
    home_defence = _shrunk_ratio(home_split.goals_against_per_game, home_table.avg_away_goals, home_split.played)

    home_xg = min(max(home_attack * away_defence * league_home_goals, XG_FLOOR), XG_CEILING)
    away_xg = min(max(away_attack * home_defence * league_away_goals, XG_FLOOR), XG_CEILING)

    return {
        "home": round(home_xg, 3),
        "away": round(away_xg, 3),
        "method": "goals-based attack/defence strength vs league average (not shot-based xG)",
        "components": {
            "home_attack": round(home_attack, 3),
            "home_defence": round(home_defence, 3),
            "away_attack": round(away_attack, 3),
            "away_defence": round(away_defence, 3),
            "league_avg_home_goals": round(league_home_goals, 3),
            "league_avg_away_goals": round(league_away_goals, 3),
        },
        "sample_games": {"home_team_home_games": home_split.played, "away_team_away_games": away_split.played},
        "cross_league": home_table.code != away_table.code,
    }


def _recent_results(team_id: int, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finished = [m for m in matches if m.get("status") == "FINISHED"]
    finished.sort(key=lambda m: m.get("utcDate") or "")
    results = []
    for match in finished[-RECENT_MATCHES:]:
        full_time = (match.get("score") or {}).get("fullTime") or {}
        home_goals, away_goals = full_time.get("home"), full_time.get("away")
        if home_goals is None or away_goals is None:
            continue
        is_home = (match.get("homeTeam") or {}).get("id") == team_id
        scored, conceded = (home_goals, away_goals) if is_home else (away_goals, home_goals)
        opponent = (match.get("awayTeam") if is_home else match.get("homeTeam")) or {}
        results.append(
            {
                "date": (match.get("utcDate") or "")[:10],
                "competition": (match.get("competition") or {}).get("name"),
                "venue": "H" if is_home else "A",
                "opponent": opponent.get("shortName") or opponent.get("name"),
                "score": f"{scored}-{conceded}",
                "result": "W" if scored > conceded else "D" if scored == conceded else "L",
            }
        )
    return results


def _team_profile(standing: TeamStanding, recent: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": standing.name,
        "competition": standing.competition_name,
        "position": standing.position,
        "points": standing.points,
        "record": {"won": standing.won, "draw": standing.draw, "lost": standing.lost},
        "overall": standing.total.to_dict(),
        "home": standing.home.to_dict() if standing.home else None,
        "away": standing.away.to_dict() if standing.away else None,
        "recent_matches": recent,
        "recent_form": "".join(r["result"] for r in recent) or standing.form,
    }


def get_match_stats(
    home_team: str,
    away_team: str,
    league: str | None = None,
    client: FootballDataClient | None = None,
) -> dict[str, Any]:
    """Aggregate statistics for a fixture.

    Keys: ``home_team``, ``away_team`` (profiles or None), ``xg_estimate``,
    ``upcoming_fixture``, ``lineups``, ``cards``, ``data_sources``, ``warnings``.
    """
    client = client or FootballDataClient()
    result: dict[str, Any] = {
        "home_team": None,
        "away_team": None,
        "xg_estimate": None,
        "upcoming_fixture": None,
        "lineups": {"available": False, "reason": "Confirmed lineups are published ~1h before kick-off and have no free API."},
        "cards": {"available": False, "reason": "Card and referee statistics have no free API source."},
        "data_sources": [],
        "warnings": [],
    }
    if not client.available:
        result["warnings"].append(
            "FOOTBALL_DATA_API_KEY is not set: no team statistics were fetched, the model uses the xG you supplied."
        )
        return result

    queries = {"home": home_team, "away": away_team}
    found: dict[str, TeamStanding | None] = {"home": None, "away": None}
    tables: dict[str, LeagueTable] = {}

    for code in _competition_search_order(league):
        if all(found.values()):
            break
        try:
            table = client.get_league_table(code)
        except FootballDataError as exc:
            result["warnings"].append(f"Standings for {code} unavailable: {exc}")
            continue
        tables[code] = table
        for side, query in queries.items():
            if found[side] is None:
                found[side] = find_team(table, query)

    today = date.today()
    for side, query in queries.items():
        standing = found[side]
        if standing is None:
            result["warnings"].append(f"'{query}' was not found in any supported league table.")
            continue
        recent: list[dict[str, Any]] = []
        try:
            matches = client.get_team_matches(standing.team_id, today - timedelta(days=90), today + timedelta(days=30))
            recent = _recent_results(standing.team_id, matches)
            if side == "home" and found["away"] is not None:
                result["upcoming_fixture"] = _find_fixture(matches, standing.team_id, found["away"].team_id)
        except FootballDataError as exc:
            result["warnings"].append(f"Recent matches for {standing.name} unavailable: {exc}")
        result[f"{side}_team"] = _team_profile(standing, recent)

    home, away = found["home"], found["away"]
    if home and away:
        estimate = estimate_expected_goals(home, tables[home.competition_code], away, tables[away.competition_code])
        if estimate is None:
            result["warnings"].append("Not enough league games played yet to estimate expected goals.")
        else:
            if estimate["cross_league"]:
                result["warnings"].append(
                    "Teams play in different domestic leagues; strengths are relative to each league and are not "
                    "directly comparable, so treat the xG estimate with extra caution."
                )
            result["xg_estimate"] = estimate

    if tables:
        result["data_sources"].append("football-data.org (standings, home/away splits, recent results)")
    return result


def _find_fixture(matches: list[dict[str, Any]], home_id: int, away_id: int) -> dict[str, Any] | None:
    for match in sorted(matches, key=lambda m: m.get("utcDate") or ""):
        if match.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        ids = {(match.get("homeTeam") or {}).get("id"), (match.get("awayTeam") or {}).get("id")}
        if ids == {home_id, away_id}:
            return {
                "kickoff_utc": match.get("utcDate"),
                "competition": (match.get("competition") or {}).get("name"),
                "listed_home_team": (match.get("homeTeam") or {}).get("name"),
                "venue_matches_input": (match.get("homeTeam") or {}).get("id") == home_id,
            }
    return None


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(BeautifulSoup(value, "html.parser").get_text(" ").split())


def _fold(text: str) -> str:
    """Lower-case and strip accents for substring matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _relevance_terms(team_name: str) -> list[str]:
    """Words a headline must contain to be about this team ("Man City" -> ["man", "city"])."""
    words = re.findall(r"[a-z0-9]+", _fold(team_name))
    terms = [w for w in words if w not in _CLUB_PREFIXES and not w.isdigit()]
    return terms or words


_CLUB_PREFIXES = {"fc", "cf", "afc", "ac", "as", "ssc", "sc", "club", "calcio", "de", "the"}


def _load_ddgs():
    try:
        from ddgs import DDGS  # renamed successor package
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None
    return DDGS


def get_team_news(team_name: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Latest news headlines about injuries, suspensions and squad issues for a team.

    Searches the past week first and widens to the past month if fewer than
    ``max_results`` distinct stories are found. Returns an empty list on failure.
    """
    ddgs_class = _load_ddgs()
    if ddgs_class is None:
        logger.error("duckduckgo-search is not installed; run pip install -r requirements.txt")
        return []

    query = f'"{team_name}" football team news injury'
    required_terms = _relevance_terms(team_name)
    collected: dict[str, dict[str, Any]] = {}
    for attempt, timelimit in enumerate(("w", "m")):
        if len(collected) >= max_results:
            break
        if attempt:
            time.sleep(1.5)  # DuckDuckGo rate-limits bursts of requests
        try:
            # record=True captures the rename warning even though duckduckgo_search resets filters to "always".
            with warnings.catch_warnings(record=True):
                raw = ddgs_class().news(query, region="wt-wt", safesearch="off", timelimit=timelimit, max_results=max_results * 3)
        except Exception as exc:  # the library raises several undocumented exception types (rate limits, timeouts)
            logger.warning("News search for %s failed: %s", team_name, exc)
            continue
        for item in raw or []:
            title = _clean_text(item.get("title"))
            key = item.get("url") or title.lower()
            if not title or key in collected:
                continue
            haystack = _fold(f"{title} {item.get('body') or ''}")
            if not all(term in haystack for term in required_terms):
                continue  # search engines return loosely related stories; keep only ones naming the team
            collected[key] = {
                "title": title,
                "source": item.get("source"),
                "published": item.get("date"),
                "url": item.get("url"),
                "snippet": _clean_text(item.get("body"))[:300],
            }

    news = sorted(collected.values(), key=lambda n: n.get("published") or "", reverse=True)
    return news[:max_results]
