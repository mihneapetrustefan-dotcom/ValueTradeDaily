# ValueTradeDaily

A command-line prediction engine for football matches. It turns expected goals into market probabilities with a Poisson model, compares them with live bookmaker prices, and uses Claude to read team news before recommending a value bet with a Kelly-sized stake.

**Competitions:** UEFA Champions League, Premier League, Serie A, Bundesliga, LaLiga, Ligue 1

**Markets:** 1X2 (home / draw / away), Over/Under 1.5, 2.5 and 3.5 goals, Both Teams To Score (yes / no)

---

## How it works

```
 team names + xG
       |
       v
 stats_scraper.get_match_stats   football-data.org: tables, home/away splits, form,
       |                         optional goals-based xG estimate (--auto-xg)
 stats_scraper.get_team_news     DuckDuckGo News: latest 5 headlines per team
       |
 poisson_model.predict           score matrix -> 1X2, O/U, BTTS probabilities
       |
 odds_api.OddsAPIClient          The Odds API: best price per outcome + margin-free
       |                         market probabilities
 llm_engine.LLMEngine            Claude reads news -> bounded xG multipliers, vetoes
       |
 betting_math                    Poisson re-run, exact EV and fractional Kelly
       v
 cli.py                          rich tables and the final recommendation
```

### The Poisson model

Each team's goals follow a Poisson distribution with mean equal to its expected goals. The probability of a score `h-a` is `P(home = h) x P(away = a)`. The engine builds this matrix up to 10-10 so that almost no probability mass is lost, renormalises it, then sums cells:

| Market | Cells summed |
|---|---|
| Home win / Draw / Away win | below / on / above the diagonal |
| Over N.5 | `h + a > N.5` |
| BTTS yes | `h >= 1` and `a >= 1` |

The 0-5 x 0-5 block is shown in the terminal.

### Value and stake sizing

For each market with a price:

- **Expected value** = `model probability x decimal odds - 1`
- **Full Kelly** = `(b x p - (1 - p)) / b`, where `b = odds - 1`
- **Recommended stake** = `full Kelly x --kelly` (default 0.25), capped at `--max-stake` (default 5% of bankroll)

A market counts as a value bet only if all of these hold:

1. The odds are real, not mock prices.
2. EV is at least `--min-edge` (default 3%).
3. The model probability is within `--max-gap` (default 15 percentage points) of the margin-free market probability. Bigger gaps almost always mean the model is missing information.
4. Claude has not vetoed the market because of team news.

### The role of Claude

Claude is given the statistics, probabilities, prices, pre-news evaluations and headlines. It returns a strict JSON assessment, enforced with structured outputs:

- a summary, key absences, confidence and an **xG multiplier** for each team (clamped to 0.85-1.15)
- **vetoed markets**, each with a reason
- a **preferred market**, a match narrative and risk flags

Python then applies the multipliers, re-runs the Poisson model and recalculates EV and Kelly stakes exactly. Stake sizes never come from LLM arithmetic. The default model is `claude-sonnet-5`; override it with `ANTHROPIC_MODEL`.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mihneapetrustefan-dotcom/ValueTradeDaily.git
cd ValueTradeDaily
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

`duckduckgo-search` has been renamed upstream to `ddgs`. The engine uses `ddgs` automatically when it is installed (`pip install ddgs`), and that package copes better with DuckDuckGo rate limits.

## Configuration

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Used for | Where to get it | Without it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | News assessment by Claude | [console.anthropic.com](https://console.anthropic.com) | No news adjustment or vetoes |
| `ODDS_API_KEY` | Live bookmaker odds | [the-odds-api.com](https://the-odds-api.com) (free tier: 500 credits/month) | Mock odds, **never** staked |
| `FOOTBALL_DATA_API_KEY` | Tables, form, `--auto-xg` | [football-data.org](https://www.football-data.org/client/register) (free tier, 10 requests/min) | Uses the xG you pass in |
| `ANTHROPIC_MODEL` (optional) | Claude model override | | `claude-sonnet-5` |

`.env` is git-ignored. Never commit real keys.

**API quota notes**
- The Odds API charges one credit per market per region. A fixture lookup costs 2 credits (h2h + totals), plus 1 for BTTS. Pass `--league` so only one competition is searched; without it the engine may search all six. Use `--no-btts-odds` to save the extra request.
- football-data.org responses are cached in `data/cache/` (standings for 6 hours, matches for 1 hour).

---

## Usage

```bash
python -m src.cli --help
python -m src.cli leagues
```

### Examples

Basic analysis with your own xG estimates:

```bash
python -m src.cli analyze "Arsenal" "Chelsea" --league epl --home-xg 1.7 --away-xg 1.1
```

Let the engine estimate xG from league tables:

```bash
python -m src.cli analyze "Inter" "Juventus" --league seriea --auto-xg
```

Champions League, with a custom bankroll and a more conservative stake:

```bash
python -m src.cli analyze "Bayern Munich" "PSG" --league ucl --home-xg 1.8 --away-xg 1.4 --bankroll 500 --kelly 0.2 --max-stake 0.03
```

Model probabilities only, with no news and no LLM:

```bash
python -m src.cli analyze "Real Madrid" "Atletico Madrid" --league laliga --no-news --no-llm
```

Save the full result as JSON (`data/predictions/`), or print the JSON for scripting:

```bash
python -m src.cli analyze "Dortmund" "Leverkusen" --league bundesliga --save
python -m src.cli analyze "Marseille" "Lyon" --league ligue1 --json
```

### Options for `analyze`

| Option | Default | Meaning |
|---|---|---|
| `--home-xg`, `--away-xg` | 1.5, 1.2 | Expected goals fed to the model |
| `--league`, `-l` | all | `ucl`, `epl`, `seriea`, `bundesliga`, `laliga`, `ligue1` |
| `--auto-xg` | off | Use the football-data.org goals-based estimate when available |
| `--bankroll` | 1000 | Converts stake percentages into amounts |
| `--kelly` | 0.25 | Fraction of full Kelly |
| `--max-stake` | 0.05 | Maximum stake per bet as a fraction of bankroll |
| `--min-edge` | 0.03 | Minimum EV to count as value |
| `--max-gap` | 0.15 | Maximum allowed difference between model and market probability |
| `--no-news` / `--no-llm` / `--no-btts-odds` | off | Skip the matching stage |
| `--save` / `--json` | off | Persist or print the JSON result |
| `--verbose`, `-v` | off | Show warnings from the data fetchers |

### Terminal output

1. Team statistics (position, goals per game, home/away splits, recent form)
2. Expected goals: estimate, value used, news multiplier, adjusted value
3. Model probabilities before and after the news, with fair odds
4. Exact-score matrix (0-5) and the most likely scores
5. Value analysis: best odds, model vs market probability, EV, Kelly stake and status for every market
6. Latest headlines per team
7. Claude's assessment: absences, narrative, risk flags, vetoes
8. **Final recommendation**: `BET` with selection, odds, EV and stake, or `NO BET` with the reason

---

## Project structure

```
ValueTradeDaily/
|-- data/                     cache and saved predictions (git-ignored contents)
|-- src/
|   |-- cli.py                typer + rich entry point
|   |-- config.py             env loading, league definitions
|   |-- fetchers/
|   |   |-- odds_api.py       The Odds API client, best prices, de-vig, mock fallback
|   |   `-- stats_scraper.py  football-data.org stats, xG estimate, DuckDuckGo news
|   |-- processing/
|   |   |-- markets.py        market ids, labels and outcome groups
|   |   `-- team_names.py     fuzzy team-name matching across providers
|   |-- models/
|   |   `-- poisson_model.py  score matrix and market probabilities
|   `-- analyzer/
|       |-- betting_math.py   EV, Kelly, market evaluation
|       `-- llm_engine.py     Claude assessment and final recommendation
|-- .env.example
`-- requirements.txt
```

## Limitations

- **The model is only as good as its xG input.** Independent Poisson is a baseline. It slightly under-predicts draws and ignores score correlation. The `--auto-xg` estimate is goals-based, not shot-based, and is shrunk towards the league average while few games have been played.
- **Cross-league matches** (Champions League) compare strengths measured in different domestic leagues, which is only a rough approximation.
- **Shot-based xG, confirmed lineups and card statistics** have no free, terms-compliant source. The engine reports them as unavailable rather than inventing them.
- **Headlines are not verified facts.** DuckDuckGo can rate-limit or return loosely related stories. The engine keeps only stories that name the team, and Claude is told to treat headlines as untrusted data.
- **Betting markets are efficient.** A positive EV number is an estimate, not a guarantee. Stakes on the same match are correlated, so do not add them up.

## Disclaimer

ValueTradeDaily is an analytical tool, not financial advice. Bet responsibly, only with money you can afford to lose, and only where betting is legal for you.
