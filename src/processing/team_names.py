"""Fuzzy team-name matching across data providers.

The Odds API says "Inter Milan", football-data.org says "FC Internazionale Milano",
a user types "inter". These helpers normalise names so they can be matched reliably.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

MATCH_THRESHOLD = 0.72

_STOPWORDS = {
    "fc", "cf", "afc", "ac", "as", "ssc", "sc", "cfc", "ogc", "club", "calcio",
    "de", "del", "the", "and", "rcd", "sad", "ud", "cd", "sv", "tsg", "vfl", "vfb",
}

# Keys and values are in normalised form (see normalize_team_name).
_ALIASES = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united",
    "inter": "internazionale milano",
    "inter milan": "internazionale milano",
    "internazionale": "internazionale milano",
    "milan": "milan",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "bayern": "bayern munchen",
    "bayern munich": "bayern munchen",
    "atletico": "atletico madrid",
    "atleti": "atletico madrid",
    "barca": "barcelona",
    "leverkusen": "bayer leverkusen",
    "gladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
    "borussia m gladbach": "borussia monchengladbach",
    "m gladbach": "borussia monchengladbach",
    "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund",
    "juve": "juventus",
    "athletic bilbao": "athletic",
    "athletic club": "athletic",
    "marseille": "olympique marseille",
    "lyon": "olympique lyonnais",
    "rennes": "stade rennais",
    "koln": "koln",
    "cologne": "koln",
}


def normalize_team_name(name: str) -> str:
    """Lower-case, strip accents/punctuation/numbers and club prefixes, then apply aliases."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in _STOPWORDS and not tok.isdigit()]
    normalized = " ".join(tokens)
    return _ALIASES.get(normalized, normalized)


def team_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two team names."""
    na, nb = normalize_team_name(a), normalize_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    tokens_a, tokens_b = set(na.split()), set(nb.split())
    # "tottenham" vs "tottenham hotspur": one name fully contained in the other.
    if tokens_a <= tokens_b or tokens_b <= tokens_a:
        return 0.9
    ratio = SequenceMatcher(None, na, nb).ratio()
    if tokens_a & tokens_b:
        # Shared words ("manchester") with different distinguishing words ("united" vs "city"):
        # judge by the distinguishing parts so spelling variants (munich/munchen) still match.
        rest_a = " ".join(sorted(tokens_a - tokens_b))
        rest_b = " ".join(sorted(tokens_b - tokens_a))
        return min(ratio, SequenceMatcher(None, rest_a, rest_b).ratio())
    return ratio


def best_match(query: str, candidates: Iterable[str], threshold: float = MATCH_THRESHOLD) -> tuple[str, float] | None:
    """Return the best-matching candidate and its score, or None below the threshold."""
    best: tuple[str, float] | None = None
    for candidate in candidates:
        score = team_similarity(query, candidate)
        if best is None or score > best[1]:
            best = (candidate, score)
    if best is None or best[1] < threshold:
        return None
    return best
