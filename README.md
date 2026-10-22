# Coors Fit

Given only a hitter's non-Coors history, can we forecast how much more or less than the league-average Coors effect he gains at Coors Field?

This is a forecast scored on hitters the model never saw, not a leaderboard of who has hit well at Coors. The hypotheses and analysis plan were committed in [docs/hypotheses.md](docs/hypotheses.md) before any model was fit. A null result is a valid outcome.

**Status:** Phases 0 to 6 of 7 are done. Main findings, which hold without 2020 and with a 400-PA trait minimum:

- **No stable Coors skill among visiting hitters.** Their true Coors effects differ by an SD of less than 27 wOBA points (upper 95% bound); the point estimate is zero.
- **The traits do not reliably beat the baseline out of sample.**
  - For unseen hitters, the per-PA gain is +154 (wOBA points)², with a 95% interval of −48 to +352.
  - For future seasons, it is −12 (−329 to +301).
- **No pre-registered trait counts as a finding.** Two traits cleared the Holm correction, but neither passes the mechanism check:
  - **Fly-ball exit velocity (H3b)** has the predicted sign, but shows no Coors gain in home runs per fly ball or in distance, so it is flagged as likely noise.
  - **Breaking-ball deficit (H1)** has the *opposite* sign to the one registered, and the effect runs through fastball outcomes rather than breaking balls.

See [Phase 6 robustness](reports/tables/phase6_robustness.md), [Phase 5 mechanisms](reports/tables/phase5_mechanisms.md), [Phase 4 results](reports/tables/phase4_results.md), the [gate](reports/tables/gate.md), the [data summary](reports/tables/data_summary.md) and the [target diagnostics](reports/tables/targets_diagnostics.md).

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

- Phase 7: the memo and the final README.
- A check that no single season drives the results (only the 2020 exclusion was run).
- Mechanism check for H4, which is outside the minimum scope.
- The pooled visitor-plus-Rockies model from the pre-registered secondary analysis (the transfer test is done).
- Weather controls (DECISIONS.md, D-12).
