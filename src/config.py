"""Central configuration: paths, environment variables and supported leagues."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PREDICTIONS_DIR = DATA_DIR / "predictions"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# Values copied verbatim from .env.example are treated as "not configured".
_PLACEHOLDER_PREFIXES = ("your_", "<", "changeme")


def get_env(name: str) -> str | None:
    """Return a configured environment variable, or None if unset or still a placeholder."""
    value = os.getenv(name, "").strip()
    if not value or value.lower().startswith(_PLACEHOLDER_PREFIXES):
        return None
    return value


def get_anthropic_model() -> str:
    return get_env("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL


@dataclass(frozen=True)
class League:
    code: str
    name: str
    odds_api_sport_key: str
    football_data_code: str
    domestic: bool


LEAGUES: dict[str, League] = {
    "ucl": League("ucl", "UEFA Champions League", "soccer_uefa_champs_league", "CL", False),
    "epl": League("epl", "Premier League", "soccer_epl", "PL", True),
    "seriea": League("seriea", "Serie A", "soccer_italy_serie_a", "SA", True),
    "bundesliga": League("bundesliga", "Bundesliga", "soccer_germany_bundesliga", "BL1", True),
    "laliga": League("laliga", "LaLiga", "soccer_spain_la_liga", "PD", True),
    "ligue1": League("ligue1", "Ligue 1", "soccer_france_ligue_one", "FL1", True),
}

_LEAGUE_ALIASES = {
    "cl": "ucl",
    "championsleague": "ucl",
    "uefachampionsleague": "ucl",
    "pl": "epl",
    "premierleague": "epl",
    "sa": "seriea",
    "bl1": "bundesliga",
    "bl": "bundesliga",
    "pd": "laliga",
    "liga": "laliga",
    "fl1": "ligue1",
    "l1": "ligue1",
}


def resolve_league(value: str) -> League:
    """Resolve a user-supplied league name/alias (case and punctuation insensitive)."""
    key = "".join(ch for ch in value.lower() if ch.isalnum())
    key = _LEAGUE_ALIASES.get(key, key)
    if key not in LEAGUES:
        valid = ", ".join(LEAGUES)
        raise ValueError(f"Unknown league '{value}'. Supported: {valid}")
    return LEAGUES[key]
