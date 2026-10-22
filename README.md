# Coors Fit

Given only a hitter's non-Coors history, can we forecast how much more or less than the league-average Coors effect he gains at Coors Field?

This is a forecast scored on hitters the model never saw, not a leaderboard of who has hit well at Coors. The hypotheses and analysis plan were committed in [docs/hypotheses.md](docs/hypotheses.md) before any model was fit. A null result is a valid outcome.

**Status:** Phases 0 to 3 of 7 are done (setup, data, targets and the gate). The gate found no stable Coors skill among visiting hitters: their true Coors effects differ by an SD of less than 27 wOBA points (upper 95% bound), and the point estimate is zero. See the [gate](reports/tables/gate.md), the [data summary](reports/tables/data_summary.md) and the [target diagnostics](reports/tables/targets_diagnostics.md). There are no prediction results yet.

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

- Phases 4 to 7: prediction, mechanism checks, robustness and write-up.
- Weather controls (DECISIONS.md, D-12).
