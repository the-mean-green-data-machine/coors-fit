# Coors Fit

Given only a hitter's non-Coors history, can we forecast how much more or less than the league-average Coors effect he gains at Coors Field?

This is a forecast scored on hitters the model never saw, not a leaderboard of who has hit well at Coors. The hypotheses and analysis plan were committed in [docs/hypotheses.md](docs/hypotheses.md) before any model was fit. A null result is a valid outcome.

**Status:** Phase 0 of 7 (setup). There are no results yet.

## Rerun

You need [uv](https://docs.astral.sh/uv/) and network access to Baseball Savant and the MLB Stats API. LightGBM also needs a system OpenMP runtime. On macOS with Homebrew, `make setup` installs it (`brew install libomp`); on Linux it is usually already present (`libgomp1`).

```bash
make all
```

`make all` installs the pinned environment from `uv.lock`, pulls the raw data and rebuilds every output. Raw pulls are cached in `data/raw/`, so a rerun only fetches what is missing. Run `make` to list the stages.

Without uv: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`, then run the stages with `UV=` overridden, or call the modules directly.

## Layout

| Path | What it holds |
|---|---|
| `docs/hypotheses.md` | Pre-registered hypotheses, tests and scoring |
| `DECISIONS.md` | Log of choices and why they were made |
| `config.yaml` | Seasons, thresholds and seed |
| `src/coorsfit/` | Pipeline code |
| `tests/` | Data sanity and leakage tests |
| `reports/` | Memo and figures |

## Not done yet

- Phases 1 to 7: ingest, targets, gate, prediction, mechanism checks, robustness and write-up.
- Weather controls (DECISIONS.md, D-12).
