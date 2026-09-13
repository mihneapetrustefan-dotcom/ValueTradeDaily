"""ValueTradeDaily command-line interface.

Usage:
    python -m src.cli analyze "Arsenal" "Chelsea" --league epl
    python -m src.cli leagues
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.analyzer.betting_math import StakeSettings
from src.analyzer.llm_engine import LLMEngine
from src.config import LEAGUES, PREDICTIONS_DIR, resolve_league
from src.fetchers.odds_api import OddsAPIClient
from src.fetchers.stats_scraper import get_match_stats, get_team_news
from src.models.poisson_model import predict
from src.processing.markets import MARKET_LABELS, MARKET_ORDER

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ValueTradeDaily: football match predictions and value-bet stake sizing.",
)
console = Console()


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}%"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@app.command()
def analyze(
    home: str = typer.Argument(..., help="Home team name, e.g. 'Arsenal'."),
    away: str = typer.Argument(..., help="Away team name, e.g. 'Chelsea'."),
    home_xg: float = typer.Option(1.5, "--home-xg", min=0.05, max=6.0, help="Expected goals for the home team."),
    away_xg: float = typer.Option(1.2, "--away-xg", min=0.05, max=6.0, help="Expected goals for the away team."),
    league: Optional[str] = typer.Option(
        None, "--league", "-l", help="ucl, epl, seriea, bundesliga, laliga or ligue1. Omit to search all (uses more API quota)."
    ),
    auto_xg: bool = typer.Option(
        False, "--auto-xg", help="Use the football-data.org goals-based xG estimate instead of --home-xg/--away-xg when available."
    ),
    bankroll: float = typer.Option(1000.0, "--bankroll", min=0.0, help="Bankroll used to convert stake fractions to amounts."),
    kelly: float = typer.Option(0.25, "--kelly", min=0.0, max=1.0, help="Fraction of full Kelly to stake."),
    max_stake: float = typer.Option(0.05, "--max-stake", min=0.0, max=1.0, help="Maximum stake per bet as a fraction of bankroll."),
    min_edge: float = typer.Option(0.03, "--min-edge", min=0.0, max=1.0, help="Minimum expected value per unit staked."),
    max_gap: float = typer.Option(
        0.15, "--max-gap", min=0.0, max=1.0, help="Reject bets where model and market probabilities differ by more than this."
    ),
    no_news: bool = typer.Option(False, "--no-news", help="Skip the DuckDuckGo news search."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the Claude news assessment."),
    no_btts_odds: bool = typer.Option(False, "--no-btts-odds", help="Skip the extra per-event BTTS odds request."),
    save: bool = typer.Option(False, "--save", help="Save the full result as JSON in data/predictions/."),
    json_output: bool = typer.Option(False, "--json", help="Print the raw JSON result instead of tables."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show warnings and debug logging."),
) -> None:
    """Predict a match, price every market and recommend a value bet with a Kelly stake."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.ERROR,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    if league:
        try:
            league = resolve_league(league).code
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--league") from exc

    settings = StakeSettings(
        bankroll=bankroll,
        kelly_multiplier=kelly,
        max_stake_fraction=max_stake,
        min_edge=min_edge,
        max_model_market_gap=max_gap,
    )
    status_console = Console(stderr=True) if json_output else console

    with status_console.status("Fetching team statistics from football-data.org..."):
        stats = get_match_stats(home, away, league)

    estimate = stats.get("xg_estimate")
    xg_notes: list[str] = []
    if auto_xg and estimate:
        used_home_xg, used_away_xg = float(estimate["home"]), float(estimate["away"])
        xg_source = "football-data.org goals-based estimate"
    else:
        used_home_xg, used_away_xg = home_xg, away_xg
        xg_source = "command line"
        if auto_xg:
            xg_notes.append("--auto-xg requested but no estimate was available; using --home-xg/--away-xg.")

    news: dict[str, list[dict[str, Any]]] = {"home": [], "away": []}
    if not no_news:
        with status_console.status(f"Searching news for {home} and {away}..."):
            news["home"] = get_team_news(home)
            time.sleep(1.5)  # spread requests to avoid DuckDuckGo rate limiting
            news["away"] = get_team_news(away)

    prediction = predict(used_home_xg, used_away_xg)

    with status_console.status("Fetching bookmaker odds..."):
        odds = OddsAPIClient().get_match_odds(home, away, league, include_btts=not no_btts_odds)

    engine = LLMEngine(enabled=not no_llm)
    label = f"Asking {engine.model} to assess the news..." if engine.available else "Computing recommendation..."
    with status_console.status(label):
        result = engine.generate_recommendation(
            home_team=home,
            away_team=away,
            home_xg=used_home_xg,
            away_xg=used_away_xg,
            stats=stats,
            prediction=prediction,
            odds=odds,
            news=news,
            settings=settings,
        )

    result["match"]["league"] = LEAGUES[league].name if league else None
    result["xg"]["source"] = xg_source
    result["xg"]["estimate"] = estimate
    result["xg"]["notes"] = xg_notes
    result["team_stats"] = stats
    result["news"] = {**news, "skipped": no_news}

    if save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = PREDICTIONS_DIR / f"{stamp}_{_slug(home)}_vs_{_slug(away)}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        status_console.print(f"[dim]Saved to {path}[/dim]")

    if json_output:
        print(json.dumps(result, indent=2, default=str))
        return

    render(result)


@app.command()
def leagues() -> None:
    """List the supported competitions."""
    table = Table(title="Supported competitions", box=box.SIMPLE_HEAVY)
    table.add_column("--league")
    table.add_column("Competition")
    table.add_column("The Odds API key", style="dim")
    table.add_column("football-data.org", style="dim")
    for lg in LEAGUES.values():
        table.add_row(lg.code, lg.name, lg.odds_api_sport_key, lg.football_data_code)
    console.print(table)


# ---------------------------------------------------------------- rendering


def render(result: dict[str, Any]) -> None:
    match = result["match"]
    stats = result["team_stats"]
    home, away = match["home_team"], match["away_team"]

    header = Text()
    header.append(f"{home}  vs  {away}\n", style="bold white")
    details = [match.get("league") or "League not specified"]
    fixture = stats.get("upcoming_fixture") or {}
    kickoff = match.get("kickoff_utc") or fixture.get("kickoff_utc")
    if kickoff:
        details.append(f"kick-off {kickoff.replace('T', ' ').replace('Z', ' UTC')}")
    header.append("  |  ".join(details), style="cyan")
    console.print(Panel(header, title="[bold green]ValueTradeDaily[/bold green]", border_style="green"))

    _render_team_stats(stats, home, away)
    _render_xg(result)
    _render_probabilities(result)
    _render_score_matrix(result, home, away)
    _render_markets(result)
    _render_news(result, home, away)
    _render_llm(result)
    _render_recommendation(result)

    warnings = list(stats.get("warnings", [])) + list(result["xg"].get("notes", []))
    if warnings:
        console.print(Panel("\n".join(f"- {w}" for w in warnings), title="Data warnings", border_style="yellow"))
    console.print(
        "[dim]Model output, not financial advice. Probabilities are estimates; bet only what you can afford to lose.[/dim]"
    )


def _render_team_stats(stats: dict[str, Any], home: str, away: str) -> None:
    home_profile, away_profile = stats.get("home_team"), stats.get("away_team")
    if not home_profile and not away_profile:
        return
    table = Table(title="Team statistics (football-data.org)", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column(home_profile["name"] if home_profile else home, justify="center")
    table.add_column(away_profile["name"] if away_profile else away, justify="center")

    def cell(profile: dict[str, Any] | None, getter) -> str:
        if not profile:
            return "-"
        try:
            value = getter(profile)
        except (KeyError, TypeError):
            return "-"
        return "-" if value is None else str(value)

    rows = [
        ("Competition", lambda p: p["competition"]),
        ("Position", lambda p: p["position"]),
        ("Points", lambda p: p["points"]),
        ("W-D-L", lambda p: f"{p['record']['won']}-{p['record']['draw']}-{p['record']['lost']}"),
        ("Goals for / game", lambda p: p["overall"]["goals_for_per_game"]),
        ("Goals against / game", lambda p: p["overall"]["goals_against_per_game"]),
        ("Home GF-GA (games)", lambda p: f"{p['home']['goals_for']}-{p['home']['goals_against']} ({p['home']['played']})"),
        ("Away GF-GA (games)", lambda p: f"{p['away']['goals_for']}-{p['away']['goals_against']} ({p['away']['played']})"),
        ("Recent form (old to new)", lambda p: p["recent_form"]),
    ]
    for name, getter in rows:
        table.add_row(name, cell(home_profile, getter), cell(away_profile, getter))
    console.print(table)


def _render_xg(result: dict[str, Any]) -> None:
    xg = result["xg"]
    table = Table(title=f"Expected goals (source: {xg['source']})", box=box.ROUNDED)
    table.add_column("")
    table.add_column("Home", justify="right")
    table.add_column("Away", justify="right")
    estimate = xg.get("estimate")
    if estimate:
        table.add_row("Goals-based estimate", f"{estimate['home']:.2f}", f"{estimate['away']:.2f}")
    table.add_row("Used by model", f"{xg['input']['home']:.2f}", f"{xg['input']['away']:.2f}")
    table.add_row("News multiplier", f"x{xg['multipliers']['home']:.2f}", f"x{xg['multipliers']['away']:.2f}")
    table.add_row("[bold]After news[/bold]", f"[bold]{xg['news_adjusted']['home']:.2f}[/bold]", f"[bold]{xg['news_adjusted']['away']:.2f}[/bold]")
    console.print(table)


def _render_probabilities(result: dict[str, Any]) -> None:
    pre = result["model_probabilities_pct"]["pre_news"]
    post = result["model_probabilities_pct"]["news_adjusted"]
    table = Table(title="Poisson model probabilities", box=box.ROUNDED)
    table.add_column("Market", style="bold")
    table.add_column("Pre-news", justify="right")
    table.add_column("After news", justify="right", style="bold")
    table.add_column("Fair odds", justify="right")
    for market in MARKET_ORDER:
        fair = result["fair_odds"].get(market)
        table.add_row(MARKET_LABELS[market], _pct(pre[market], 2), _pct(post[market], 2), f"{fair:.2f}" if fair else "-")
    console.print(table)


def _render_score_matrix(result: dict[str, Any], home: str, away: str) -> None:
    grid = result["score_matrix"]["percentages"]
    table = Table(title=f"Exact score probabilities (%), rows = {home} goals, columns = {away} goals", box=box.MINIMAL_HEAVY_HEAD)
    table.add_column("H \\ A", style="bold", justify="center")
    for away_goals in range(len(grid[0])):
        table.add_column(str(away_goals), justify="right")
    peak = max(max(row) for row in grid)
    for home_goals, row in enumerate(grid):
        cells = []
        for value in row:
            style = "bold green" if value == peak else "green" if value >= peak * 0.6 else ""
            cells.append(f"[{style}]{value:.1f}[/{style}]" if style else f"{value:.1f}")
        table.add_row(str(home_goals), *cells)
    console.print(table)
    top = ", ".join(f"{s['score']} ({s['percentage']:.1f}%)" for s in result["most_likely_scores"])
    console.print(f"[dim]Most likely scores: {top}[/dim]")


def _render_markets(result: dict[str, Any]) -> None:
    odds = result["odds"]
    title = f"Value analysis ({odds['source']}, {odds['bookmakers']} bookmaker(s))"
    border = "red" if odds["is_mock"] else "blue"
    wide = console.width >= 110  # narrow terminals drop the bookmaker and full-Kelly columns
    table = Table(title=title, box=box.ROUNDED if wide else box.SIMPLE_HEAD, border_style=border)
    table.add_column("Market", style="bold", no_wrap=True, min_width=15)
    table.add_column("Odds", justify="right", no_wrap=True, min_width=5)
    if wide:
        table.add_column("Bookmaker", overflow="ellipsis", no_wrap=True, max_width=18)
    table.add_column("Model", justify="right", no_wrap=True, min_width=6)
    table.add_column("Mkt fair", justify="right", no_wrap=True, min_width=8)
    table.add_column("EV", justify="right", no_wrap=True, min_width=7)
    if wide:
        table.add_column("Full Kelly", justify="right", no_wrap=True, min_width=10)
    table.add_column("Stake", justify="right", no_wrap=True, min_width=6)
    table.add_column("Status", overflow="ellipsis", no_wrap=True)

    for evaluation in result["market_evaluations"]:
        ev = evaluation["expected_value"]
        ev_text = "-" if ev is None else f"{ev * 100:+.1f}%"
        ev_style = "green" if ev is not None and ev > 0 else "red" if ev is not None else ""
        status_style = "bold green" if evaluation["is_value"] else "dim"
        if not evaluation["is_value"]:
            stake = "-"
        elif wide:
            stake = f"{evaluation['stake_fraction'] * 100:.2f}% ({evaluation['stake_amount']:.2f})"
        else:
            stake = f"{evaluation['stake_fraction'] * 100:.2f}%"
        fair = evaluation["market_fair_probability"]
        cells: list[Any] = [evaluation["label"], f"{evaluation['best_odds']:.2f}" if evaluation["best_odds"] else "-"]
        if wide:
            cells.append(evaluation["bookmaker"] or "-")
        cells += [
            _pct(evaluation["model_probability"] * 100),
            _pct(fair * 100 if fair is not None else None),
            Text(ev_text, style=ev_style),
        ]
        if wide:
            cells.append(_pct(evaluation["kelly_full"] * 100))
        cells += [stake, Text(_short_status(evaluation), style=status_style)]
        table.add_row(*cells)
    console.print(table)
    if odds["is_mock"]:
        console.print("[bold red]MOCK ODDS: prices are placeholders, not bookmaker quotes. No stakes are sized.[/bold red]")
    notes = list(odds.get("notes") or [])
    if odds.get("requests_remaining"):
        notes.append(f"The Odds API credits remaining: {odds['requests_remaining']}")
    if notes:
        console.print(Panel("\n".join(notes), border_style=border, title="Odds notes"))


def _short_status(evaluation: dict[str, Any]) -> str:
    reason = evaluation["reason"]
    if evaluation["is_value"]:
        return "VALUE"
    if reason.startswith("mock"):
        return "mock"
    if reason.startswith("EV below"):
        return "low EV"
    if reason.startswith("model differs"):
        return "market gap"
    if reason.startswith("vetoed"):
        return "vetoed"
    if reason == "no odds available":
        return "no odds"
    return reason


def _render_news(result: dict[str, Any], home: str, away: str) -> None:
    news = result["news"]
    if news.get("skipped"):
        return
    blocks = []
    for side, team in (("home", home), ("away", away)):
        text = Text()
        text.append(f"{team}\n", style="bold")
        if not news[side]:
            text.append("  No relevant headlines found (DuckDuckGo may be rate-limiting; run with -v for details).\n", style="dim")
        for item in news[side]:
            published = (item.get("published") or "")[:10]
            text.append(f"  - {item['title']}", style="white")
            text.append(f"  [{item.get('source') or '?'} {published}]\n", style="dim")
        blocks.append(text)
    console.print(Panel(Group(*blocks), title="Latest team news (DuckDuckGo)", border_style="magenta"))


def _render_llm(result: dict[str, Any]) -> None:
    analysis = result["llm_analysis"]
    meta = result["llm"]
    if not meta["used"]:
        console.print(Panel(analysis["match_narrative"] or "LLM analysis not run.", title="LLM analysis", border_style="dim"))
        return

    body = Text()
    for side in ("home", "away"):
        impact = analysis["news_impact"][side]
        team = result["match"][f"{side}_team"]
        body.append(f"{team}", style="bold")
        body.append(f"  xG x{impact['xg_multiplier']:.2f}, confidence {impact['confidence']}\n", style="cyan")
        body.append(f"  {impact['summary']}\n")
        if impact["key_absences"]:
            body.append(f"  Absences: {', '.join(impact['key_absences'])}\n", style="yellow")
    body.append("\n")
    body.append(analysis["match_narrative"] + "\n")
    if analysis["risk_flags"]:
        body.append("\nRisk flags:\n", style="bold red")
        for flag in analysis["risk_flags"]:
            body.append(f"  - {flag}\n", style="red")
    if analysis["vetoed_markets"]:
        body.append("\nVetoed markets:\n", style="bold yellow")
        for veto in analysis["vetoed_markets"]:
            body.append(f"  - {MARKET_LABELS[veto['market']]}: {veto['reason']}\n", style="yellow")
    console.print(Panel(body, title=f"LLM analysis ({meta['model']})", border_style="cyan"))


def _render_recommendation(result: dict[str, Any]) -> None:
    rec = result["recommendation"]
    if rec["verdict"] != "BET":
        console.print(
            Panel(
                Text(f"NO BET\n\n{rec['reason']}", style="bold yellow"),
                title="Final recommendation",
                border_style="yellow",
            )
        )
        return

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Selection", f"[bold green]{rec['label']}[/bold green]")
    table.add_row("Best odds", f"{rec['odds']:.2f} at {rec['bookmaker']}")
    table.add_row("Model probability", _pct(rec["model_probability_pct"]))
    table.add_row("Market fair probability", _pct(rec["market_fair_probability_pct"]))
    table.add_row("Expected value", f"{rec['expected_value_pct']:+.1f}%")
    table.add_row("Full Kelly", _pct(rec["kelly_full_pct"], 2))
    settings = result["settings"]
    table.add_row(
        "Recommended stake",
        f"[bold]{rec['stake_pct_of_bankroll']:.2f}% of bankroll = {rec['stake_amount']:.2f}[/bold] "
        f"({settings['kelly_multiplier']:g}x Kelly, cap {settings['max_stake_fraction']:.0%})",
    )
    table.add_row("Confidence", rec["confidence"])
    parts: list[Any] = [table, Text("\n" + rec["rationale"])]
    if result["alternatives"]:
        alt = ", ".join(f"{a['label']} @ {a['odds']:.2f} (EV {a['expected_value_pct']:+.1f}%)" for a in result["alternatives"])
        parts.append(Text(f"\nOther value: {alt}", style="dim"))
    console.print(Panel(Group(*parts), title="Final recommendation: BET", border_style="bold green"))


if __name__ == "__main__":
    app()
