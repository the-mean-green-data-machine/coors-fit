# Decisions

A running log of choices and the reasons for them, newest phase last. Numbers quoted here come from the environment checks run before Phase 0. From Phase 1 on, every number comes from code in this repo.

## Phase 0: setup (2026-10-21)

### D-01 · Python environment
- **Decision:** Python 3.12, managed with uv. Direct dependencies are pinned exactly in `pyproject.toml`, and every transitive dependency is locked in `uv.lock`. `requirements.txt` is exported from the lock for pip users (`make requirements`).
- **Why:** `make all` has to rebuild the same environment on any machine. pybaseball 2.2.7 is the last release (2023); it pulled 2015 to 2026 data correctly with pandas 3.0.6 in testing.

### D-02 · LightGBM, with setup installing its OpenMP runtime
- **Decision:** M2 uses LightGBM 4.7.0, as the plan specifies. `make setup`, which `make all` runs first, checks that LightGBM loads. If it doesn't, setup installs the OpenMP runtime: on macOS with `brew install libomp`, and elsewhere it prints the package to install and stops.
- **Why:** the LightGBM wheel links against a system OpenMP runtime that pip cannot install. It failed to load on a clean macOS machine until Homebrew's `libomp` was installed. Handling this in `make setup` keeps the rebuild free of manual steps on macOS. On Linux, the runtime (`libgomp`) is usually already present.
- **Considered:** switching M2 to scikit-learn's `HistGradientBoostingRegressor`, which needs no system library. Rejected because we want LightGBM available in this project and generally.

### D-03 · Ingest: serial, one day per request, cached, retried
- **Decision:** pull one day per request, one request at a time. Each day is cached to parquet, failures are retried with exponential backoff (6 tries, 2 to 60 seconds), and a rerun skips days already cached.
- **Why:**
  - Pulling serially is fast enough: the full 2025 season (712,528 pitches) took 50 seconds.
  - pybaseball's default parallel, uncached pull died partway through a season on a transient network drop. The same drop also hit google.com, so it was not Savant blocking us.
  - A two-week 2015 pull hit two drops and recovered through retries.

### D-04 · Data examined before pre-registration
Everything looked at before `docs/hypotheses.md` was committed:
- **Fields:** `woba_value` and `woba_denom` are on every PA-ending row.
  - `woba_denom` is 0 for sac bunts and IBB, and missing for `truncated_pa`.
  - `estimated_woba_using_speedangle` is also filled for strikeouts, walks and HBP, so batted-ball checks must filter on `type == 'X'`.
  - Savant's per-pitch run value (`delta_run_exp`) is present on 99.5% of 2015 pitches.
  - Sprint speed is available from the Savant leaderboard for 2015 to 2026.
  - Statcast has no weather fields.
- **Missing tracking on batted balls:**
  - May 2015: exit velocity and launch angle 3.9% missing, distance 13.8%, expected wOBA 4.0%.
  - 2025: all under 1%.
- **2025 sizing:**
  - 181,317 wOBA PAs, with a per-PA wOBA SD of 0.514.
  - 3,337 visiting Coors PAs from 289 hitters. Coors PAs per visiting hitter: median 11, 90th percentile 19.
  - Visitors' wOBA was .391 at Coors against .315 on the road elsewhere.
  - Rockies hitters: 2,990 Coors PAs from 30 hitters, 68% of them from the 9 with at least 100 road PAs. Their wOBA was .340 at home and .268 on the road.
- **2025 breaking-ball movement** (SL, CU, KC, SV, ST): 6.1 in. horizontal and 4.2 in. vertical at Coors, against 8.1 and 5.5 elsewhere.
- **No trait was compared with any Coors outcome.**

### D-05 · Season range
- **Decision:** pull the 2015 to 2026 regular seasons. Target seasons are 2016 to 2026; 2015 only feeds traits.
- **Why:**
  - Pulls are cheap (D-03), and statistical power is the binding constraint, so more seasons is the main way to get more of it.
  - 2015 has the largest tracking gaps (D-04), so it is used only for traits.
  - The 2026 regular season is complete.
  - 2016 targets have only one prior season for traits. Shrinkage accounts for the thinner sample.

### D-06 · Coors means the venue, not the home team
- **Decision:** a PA is at Coors when the game's MLB Stats API venue is 19 (Coors Field).
- **Why:** the Rockies were the home team for games 746560 and 746561 (April 27–28, 2024) at Estadio Alfredo Harp Helú in Mexico City. These are the only regular-season Rockies home games away from Coors in 2015 to 2026. `home_team == 'COL'` alone would count them as Coors.

### D-07 · Visitors are primary; Rockies hitters are a pre-registered secondary analysis
- **Decision:** the headline result and the five confirmatory tests use visiting hitters only. Rockies hitters enter a pooled model and a transfer test (train on visitors, predict Rockies hitters, scored by calibration slope), always reported separately. This is a change from the v2 plan, which pooled both groups in the primary model.
- **Why:**
  - A Rockies hitter's traits and baseline come from road games that altitude may distort: in 2025 Rockies hitters had a .268 road wOBA against .318 for road hitters league-wide.
  - About a dozen regulars per season would carry half of all Coors PAs.
  - Road struggles against breaking balls would raise both the H1 trait and the Coors residual, so pooling could confirm H1 for the wrong reason.
  - Trait-by-`home_player` interactions were dropped from the minimum version: five interactions estimated from a few dozen Rockies hitters would shrink to about zero and tell us little, and the transfer test answers the same question more cleanly.

### D-08 · Eligibility thresholds
| Rule | Value | Reason |
|---|---|---|
| Prior non-Coors PAs, two seasons pooled | ≥ 200 (sensitivity check: 400) | Gives usable trait estimates, and also screens out most pre-2022 pitchers batting. |
| Same-season non-Coors PAs | ≥ 100 | The expected level has to be estimable. In 2025 this drops about the bottom 10% of Coors visitors, mostly call-ups. |
| Coors PAs | No minimum | Shrinkage handles small samples. A 15-PA minimum would have kept only 36% of 2025 visiting Coors PAs. |
| Coors PAs for per-hitter displays | ≥ 10 | For readability only; does not affect the model or scoring. |

### D-09 · Phase 3 gate: estimate how much hitters truly differ
- **Decision:** the gate is τ, the between-hitter SD of the true Coors effect, pooled across seasons, estimated by method of moments with a bootstrap by batter. A career-level odd/even-season split is secondary.
- **Why:** a median visitor-season of 11 Coors PAs has a residual SE of about 0.155. Even if hitters truly differed by an SD of 0.020, a single hitter-season's reliability would be only about 0.016, so the original within-season split-half check would read near zero whatever the truth. The target is unchanged; only the way its signal is measured changed, decided before any results.

### D-10 · Hypotheses changed from the original plan
Changed before pre-registration:
- **H1** now predicts that hitters who are *weaker* against breaking balls gain more. Flatter breaking balls shrink a weakness, so hitters who already handle them have less to gain. The trait is fastball minus breaking-ball run value per 100 pitches.
- **H2** uses line-drive rate instead of ground-ball rate. Ground-ball rate and fly-ball rate are near complements, which would set H2 and H3 against each other, and Coors's infield is regulation size.
- **H3** is split into fly-ball rate (H3a) and fly-ball exit velocity (H3b).
- **H4** is labeled the positive control, since in additive wOBA points it is close to mechanical.

### D-11 · Estimation and scoring
- **Coefficient tests** use the raw hitter-season residual, weighted by Coors PAs. This is equivalent to a PA-level model with trait-by-Coors interactions and the expectation as an offset. Shrinking the target first would bias the coefficients, so empirical-Bayes shrinkage is used only for per-hitter estimates, the gate and displays.
- **Five confirmatory tests,** Holm-corrected, with a bootstrap by batter.
- **Lift** is the per-PA reduction in squared error relative to B0 on held-out visiting Coors PAs, reported next to the τ² ceiling.
- **Planning estimate of the minimum detectable effect:** about 9 wOBA points per trait SD, or about 11 after correcting for five tests. It is recomputed from actual eligible counts in Phase 2.

### D-12 · Weather out of the minimum version
- **Decision:** no weather controls in the minimum version; listed under "Not done yet."
- **Why:** Statcast has no weather fields. The MLB Stats API game feed has temperature, wind and conditions for each game (checked on a 2025 Coors game), but adding it means a second pull, and weather is a game-level control that does not bear on the trait-by-Coors effects being tested.

### D-13 · Seeds
- **Decision:** one seed (`config.yaml: seed`). Each random step takes its own NumPy generator seeded with `[seed, stream]` (`coorsfit.config.rng`), so adding a step does not shift the random numbers of the others. `set_global_seeds()` covers libraries that read global state.
