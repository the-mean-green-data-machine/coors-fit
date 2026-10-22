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
  - Pulling serially is fast enough: the full 2025 season (712,528 pitches) took 50 seconds. *Corrected in D-14: those days were mostly cached by Savant's CDN; an uncached day takes about 5 seconds, so the pull now runs 4 requests at a time.*
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

## Phase 1: ingest and clean (2026-10-21)

### D-14 · Pull four days at a time
- **Decision:** ingest sends 4 day requests at once (`ingest.workers`), each with the D-03 retries.
- **Why:** the 50-second 2025 timing in D-03 came mostly from cached responses. An uncached day takes Savant about 5 seconds, so a serial pull would take about 3 hours. With 4 workers the full 2015–2026 pull took about 34 minutes. Retries absorbed 145 network drops, and no day failed. 2024–2026 came back in about 12 seconds per season, presumably from Savant's cache; the schedule check (D-15) and the sanity tests confirm those days are complete.

### D-15 · Pull the dates of played games, and check each day against the schedule
- **Decision:** ingest pulls exactly the dates of played regular-season games from the MLB Stats API schedule (`codedGameState == 'F'`) rather than a calendar range. A day whose response holds none of its scheduled games is retried instead of being cached as empty.
- **Why:** a transient failure can return an empty frame, which would otherwise be cached as a day without games. The result: all 27,622 played games from 2015 to 2026 are in Statcast, and no Statcast game is missing from the schedule.

### D-16 · Compute the wOBA denominator from the event
- **Decision:** `woba_denom` is 0 for `sac_bunt`, `sac_bunt_double_play`, `intent_walk` and `truncated_pa`, and 1 otherwise. Savant's column is kept as `woba_denom_savant`.
- **Why:**
  - Savant leaves `woba_denom` blank on ordinary PAs: 3.0% in 2015, falling to 0.5–0.6% from 2021. `woba_value` is present on all of them.
  - The rule matches Savant on every PA where Savant fills it, in every season.
  - Savant counts catcher's interference (denominator 1, value 0.7), unlike the textbook formula. The rule follows Savant, because `woba_value` comes from Savant.
  - PAs with a blank Savant denominator have the same mean `woba_value` per event as other PAs, within 0.01.

### D-17 · Cleaning rules
- **Duplicates:** pitches are keyed on `game_pk`, `at_bat_number` and `pitch_number`. No season had a duplicate pitch, but the deduplication stays as a guard.
- **PAs:** a PA is the pitch row that carries an event. If a PA had two event rows, the later pitch would be kept.
- **Types:** columns are cast through nullable types, because raw days disagree on dtype. A missing ID or count raises an error rather than being dropped.
- **Expected wOBA:** `xwoba_bip` keeps Savant's expected wOBA only on batted balls.
- **Venue:** comes from the schedule. Every pitch matched a played game.

### D-18 · Storage
- **Raw (1.3 GB):** `data/raw/statcast/{season}/{date}.parquet`, plus `schedule/` and `sprint_speed/` for each season.
- **Interim (238 MB):** `data/interim/pitches_{season}.parquet`.
- **Processed (80 MB):** `data/processed/pa.parquet`, with 2,092,114 PAs.
- **What's committed:** none of the data. `reports/tables/data_summary.md` is committed, so readers can see the counts without rebuilding.

### Observations from the data summary (no decision)
- Visitors' wOBA at Coors exceeded visitors' road wOBA at other parks by between .017 (2021) and .075 (2025). A swing that large between seasons fits changes in Rockies pitching quality, which is why the Phase 2 expectation controls for the pitchers faced.
- Visitors had 33,415 Coors wOBA PAs in the 2016–2026 target seasons, close to the 35,000 planning figure in `docs/hypotheses.md`. That count is before the eligibility thresholds.

## Phase 2: outcome and target (2026-10-21)

### D-19 · Expectation model
- **Decision:** expected wOBA for a PA is the league mean plus effects for batter, pitcher, park, platoon cell (batter side × pitcher hand) and home/road.
  - It is fit separately for each season on non-Coors wOBA PAs.
  - Batter, pitcher and park effects are empirical-Bayes shrunk toward zero; platoon and home/road are estimated directly.
  - It is fit by backfitting, converging in 18 to 42 iterations and about 0.1 s per season.
  - A Coors PA gets a park effect of zero, so its residual compares Coors with the average non-Coors park.
- **Why:**
  - **Home/road term:** a visitor bats as the road team at Coors. The estimated home advantage is 0.006 to 0.018 wOBA depending on the season.
  - **Park term:** a hitter's non-Coors level otherwise carries his home park's effect.
  - **A leakage test** shows changing Coors outcomes never changes an expected value.
- **Pre-registration:** this is amendment A-1 in `docs/hypotheses.md`.

### D-20 · Shrinkage corrected for PAs within a game being correlated
- **Decision:** before shrinking each term, inflate the noise variance by a design effect. It compares the spread of each batter's, pitcher's or park's per-game totals with what independent PAs would give.
- **Why:**
  - Out-of-sample checks (fit on odd games, score even games) showed the pitcher term under-shrunk, with coefficients of 0.55 to 0.75 against a calibrated 1.
  - With the correction, the overall held-out slope averages 0.92 across seasons (from 0.69 in 2020 to 1.03), and the batter term averages about 1.07.
  - The pitcher term is still somewhat over-dispersed (about 0.85).
- **Effect on the targets:** small. For a hitter 40 points above average, an 8% overspread moves his expected value by about 3 points, against a minimum detectable effect of about 9. Pitcher miscalibration mostly shifts every visitor in a season together, and the season's Coors effect absorbs that.
- **Considered:** estimating each term's spread from split halves. Not done: the current calibration is adequate for a target this noisy, and the time is better spent on Phases 3 and 4.

### D-21 · Pitchers' own PAs left out of the expectation fit
- **Decision:** a player who threw at least 200 pitches in a season and batted fewer than 100 times is treated as a pitcher batting. Those PAs are left out of the fit.
- **Why:** before 2022 there were about 5,000 such PAs per season. Their very low wOBA inflated the estimated spread between batters, so regular hitters were shrunk too little; in a half-season test for 2019, the held-out slope rose from 0.77 to 0.80 once they were dropped. Two-way players who bat regularly are not affected (unit test). Fewer than 10 PAs per season are flagged from 2022 on.

### D-22 · Hitter-season targets and their shrinkage
- **Decision:** each hitter-season's deviation is measured from its group's Coors effect in that season, then shrunk with a spread between hitters estimated by method of moments.
  - Groups are visitors and Rockies hitters.
  - Using the season's effect keeps season-to-season swings, such as a better or worse Rockies staff, from counting as hitter skill.
  - The league-average effect used for prediction is refit inside each Phase 4 training fold.
- **Results:**
  - **Visitors:** 1,877 eligible hitter-seasons, with a median of 12 Coors PAs each. The spread between hitters at the hitter-season level is estimated at zero, and 96.2% of hitter-seasons fall inside the band noise alone would produce. Every visitor hitter-season therefore shrinks to its season's effect. This is the situation D-09 anticipated; Phase 3 pools each hitter's seasons.
  - **Rockies hitters:** 89 eligible hitter-seasons, with a median of 236 Coors PAs each. The spread between hitters is estimated at 0.027, and the median shrink factor is 0.36. Road-based baselines may inflate that spread (D-07).
  - **Pooled Coors effect:** +0.041 for visitors and +0.056 for Rockies hitters. By season, visitors range from +0.012 (2021) to +0.065 (2025), with standard errors near 0.01.

### D-23 · Open question for Phase 4: Coors residuals slope with hitter quality
- **Finding:** among visitors, the Coors residual (net of the season effect) slopes against expected wOBA at −0.28 ± 0.12, clustered by batter. By quintile of expected wOBA it runs from +0.016 for the weakest to −0.009 for the strongest, but not monotonically.
- **Comparison:** the expectation's own out-of-sample miscalibration would give about −0.08. The excess (about −0.20 ± 0.12) is roughly 1.7 standard errors.
- **Rockies hitters** show no gradient (−0.03 ± 0.20).
- **Why it matters:** several traits correlate with hitter quality (fly-ball exit velocity, strikeout rate). Left alone, the gradient could bias their tests by about a third of the minimum detectable effect.
- **Status:** a decision for the user before any trait is computed. It is not yet a pre-registration amendment.

### D-24 · Quality control adopted (resolves D-23)
- **Decision (user, 2026-10-21):** amend the pre-registration as A-3.
  - The confirmatory fit, M1 and M2 control for each hitter-season's centered expected wOBA at Coors.
  - The headline lift compares M1 with B0q, a baseline with the same quality slope fit on the training fold.
  - The pre-registered B0 and the uncontrolled fits are reported as sensitivities.
- **Why:** the gradient could otherwise be credited to traits that track hitter quality. Putting the same slope in the baseline means any lift comes from the traits alone.
- **Committed** on its own, before any Phase 3 or Phase 4 code.

## Phase 3: the gate (2026-10-21)

### D-25 · The gate's noise term (amendment A-4)
- **Decision:** for each hitter, the noise subtracted from his pooled mean residual has three parts:
  - his own per-PA variance divided by his Coors PAs,
  - multiplied by the game design effect,
  - plus the variance of his expectation error, combined across seasons.
  Phase 2's hitter-season shrinkage also removes the expectation error. It keeps a pooled per-PA variance there, because it only feeds displays.
- **Why:**
  - The batter-effect estimate's error has an SD of 0.019 for visitors and 0.022 for Rockies hitters. It is shared within a season, about as large as the effect being tested, and would otherwise count as skill. Removing it cut Rockies hitters' hitter-season τ from 0.027 to 0.016.
  - Per-PA variance rises with Coors PAs: the correlation is 0.12 for visitors and 0.52 for Rockies hitters. A pooled variance overstated the noise; at the hitter-season level, visitors' τ² moved from −0.0025 to −0.0015 with each hitter's own variance.
- **Tests:** the estimator recovers a known spread (τ = 0.04). It does not pass on pure noise. Ignoring expectation error inflates τ².

### D-26 · Gate result: no stable Coors skill detected
- **Visitors (primary):** 708 hitters, 25,275 Coors PAs, median 23 per hitter.
  - τ² = −0.00013, with a 95% interval of −0.00095 to +0.00070.
  - Hitters' true Coors effects differ by an SD of less than 27 wOBA points (upper bound); the point estimate is 0.
  - Net of the hitter-quality slope (−0.180 ± 0.162 at the hitter-season level), the result is the same.
- **Odd vs even seasons:** r = −0.02 (−0.25 to +0.22) for the 77 visitors with at least 30 Coors PAs in each half.
- **Rockies hitters:** 45 hitters with a median of 248 Coors PAs each. τ = 19 wOBA points, but the τ² interval (−0.00009 to +0.00081) includes zero.
- **What it means for Phase 4:**
  - A perfect forecast of each hitter's true effect could cut the per-PA squared error by at most τ², which is at most 0.0007, about 0.2% of the per-PA variance of 0.30.
  - The data cannot rule out an SD of up to 27 points, so trait effects remain testable. Phase 4 runs as pre-registered.

## Phase 4: prediction and evaluation (2026-10-21)

### D-27 · Trait construction
- **Definitions.** All traits use non-Coors data from the two seasons before the target season:
  - **K rate:** strikeouts per PA, excluding truncated PAs.
  - **Line-drive and fly-ball rates:** per batted ball with a type. Pop-ups count in the denominator but not as fly balls.
  - **Fly-ball exit velocity:** the mean over fly balls with a measured exit velocity.
  - **`brk_deficit`:** run value per 100 pitches against FF, SI and FC, minus against SL, ST, SV, CU, KC and CS.
  - **Sprint speed:** the most recent season in the window with at least 10 competitive runs. It is not shrunk, because it is a measurement rather than a noisy rate. It is available for every eligible hitter-season.
- **Shrinkage and scaling.** Each target season is handled separately. Each trait is shrunk toward the mean of that season's population by the share of spread that is not sampling noise, then standardized within the season. The population is eligible hitters (at least 200 prior and 100 same-season non-Coors PAs): 3,539 hitter-seasons in all.
- **A leak the tests caught.** League rates and pooled within-batter variances were at first computed across every target season's window at once, so later seasons touched an earlier season's shrinkage (traits moved by about 0.1%). They are now computed per target season. The leakage tests on synthetic data and the rebuild of real 2022 traits now pass: changing Coors rows, the target season or later seasons leaves the traits unchanged.

### D-28 · Model details
- **Fitting.** Weighted least squares on hitter-seasons (weight = Coors PAs) equals least squares on PAs; a unit test checks this.
- **M1.** Ridge penalizes only the traits. The penalty is chosen by five-fold CV grouped by batter inside each training set; the chosen values were 1,000 to 10,000, median 3,000.
- **M2.** LightGBM is untuned: 300 trees, learning rate 0.02, 7 leaves, at least 50 rows per leaf.
- **Not done.** The pooled visitor-plus-Rockies model from the pre-registered secondary analysis was not built. It is listed under "Not done yet".

### D-29 · Phase 4 results
Full tables are in `reports/tables/phase4_results.md`.

- **Out-of-sample lift (headline):** M1 vs B0q, in (wOBA points)² per PA, with a ceiling of 704.

  | Scheme | Lift | 95% interval | RMSE ratio | Share of ceiling |
  |---|---:|---|---:|---:|
  | Grouped CV (unseen hitters) | +154 | −48 to +352 | 0.99974 | 22% |
  | Rolling origin (future seasons) | −12 | −329 to +301 | — | — |

  Both intervals include zero.
- **The other comparisons do not help either:**
  - **M2 does worse than B0q:** −368 (−745 to +1) under grouped CV and −700 (−1,255 to −163) under rolling origin. The boosted model overfits noise.
  - **B0q vs B0:** −34 and −31. The quality slope alone does not improve forecasts.
  - **M1u vs B0:** +99 and +30, both with intervals that include zero.
- **Confirmatory tests** (primary specification, n = 1,877 hitter-seasons, wOBA points per trait SD):

  | Hypothesis | Trait | Estimate | 95% interval | Holm p | Reading |
  |---|---|---:|---|---:|---|
  | H1 | `brk_deficit` | −9.6 | −15.6 to −3.4 | 0.008 | Wrong sign, so it contradicts H1 as registered |
  | H2 | `ld_rate` | −3.6 | −10.3 to +3.1 | — | Null |
  | H3a | `fb_rate` | +1.8 | −4.2 to +8.2 | — | Null |
  | H3b | `fb_ev` | +12.3 | +2.8 to +21.6 | 0.040 | Predicted sign and survives Holm; a finding only if the Phase 5 mechanism check agrees |
  | H4 | `k_rate` | −2.2 | −11.3 to +6.7 | — | Null |

  - The intercept, the visitors' Coors effect, is +40.5 (33.6 to 46.9).
  - The quality coefficient is −360 (−705 to −15) per unit of expected wOBA.
- **H1 runs opposite to the direction I proposed.** Hitters who are relatively weaker against breaking balls gain *less* at Coors. The original plan's wording ("hitters who handle breaking pitches well gain more") matches that sign. It cannot be claimed as confirmatory now, because the registered direction was the reverse. It is reported as contradicting H1, and the reversal is noted for future work.
- **H3b depends on the quality control.** Without quality (sensitivity), H3b is +8.1 (−0.3 to +16.7), Holm p = 0.26, and does not survive. H1 survives either way.
- **Exploratory, H5:** sprint speed is +10.6 (3.2 to 17.6). Being exploratory, it carries no confirmatory weight.
- **Coverage:** held-out 50% intervals cover 51–52% and 90% intervals cover 92–94%.
- **Rockies transfer:** calibration slope +0.45 (−0.21 to +1.24) across 45 hitters. Inconclusive.

## Phase 5: mechanism checks (2026-10-21)

### D-30 · How mechanism checks are scored (written and committed before running them)
- **Design.**
  - For each eligible visitor hitter-season, take the mechanism outcome at Coors minus the same hitter's outcome at other parks in the same season. Only his visiting PAs at Coors count.
  - Regress that difference on the confirmatory specification: the five traits plus `quality`.
  - Weight each hitter-season by the inverse variance of its difference, n_Coors · n_other / (n_Coors + n_other).
  - Intervals come from 2,000 bootstrap draws over batters. There is no multiplicity correction, because the mechanism checks support the confirmatory tests rather than replace them.
- **Outcomes** (minimum version: H1 to H3):
  - **H1:** run value per 100 pitches against breaking balls, and the same against fastballs.
  - **H2:** hit rate on in-park line drives and fly balls (home runs excluded), and home runs per PA.
  - **H3a and H3b:** home runs per fly ball, fly-ball distance, and actual minus expected wOBA on fly balls.
- **When a mechanism "agrees":**
  - **H3a and H3b:** the trait's coefficient is positive, with a 95% interval above zero, for at least one of the three fly-ball outcomes, and none of them is significantly negative.
  - **H2:** the coefficient of `ld_rate` on the in-park hit rate is positive with an interval above zero, and its coefficient on home runs per PA is not significantly positive.
  - **H1 (registered +):** the coefficient of `brk_deficit` on breaking-ball run value is positive with an interval above zero, and larger than its coefficient on fastball run value.
    - The confirmatory result has the opposite sign, so H1 is contradicted whatever this check shows.
    - The check is still reported, to say whether the observed negative effect runs through breaking balls: a negative breaking-ball coefficient below the fastball coefficient.
- **Verdict:** a trait is a finding only if it passed the confirmatory test (D-29) and its mechanism agrees. A trait that passed without mechanism evidence is flagged as likely noise.

### D-31 · Phase 5 results
Full tables are in `reports/tables/phase5_mechanisms.md`.

- **Sanity check: the Coors effect on each outcome.** These are visitors at Coors minus the same hitters elsewhere, weighted averages:

  | Outcome | Coors minus elsewhere |
  |---|---|
  | Fly-ball distance | +16.0 ft (consistent with the known ~5% carry boost) |
  | Home runs per fly ball | +2.2 percentage points |
  | wOBA minus expected wOBA on fly balls | +87 points |
  | Hit rate on in-park line drives and fly balls | +5.9 percentage points |
  | Home runs per PA | +0.36 percentage points |
  | Run value against fastballs | +1.08 per 100 pitches |
  | Run value against breaking balls | +0.38 per 100 pitches |

- **Each hypothesis's own trait** (coefficient per trait SD, 95% interval):

  | Hypothesis | Outcome | Estimate | 95% interval | Unit |
  |---|---|---:|---|---|
  | H3b (`fb_ev`) | Home runs per fly ball | +0.03 | −1.58 to +1.65 | pp |
  | H3b (`fb_ev`) | Fly-ball distance | +0.6 | −2.4 to +3.5 | ft |
  | H3b (`fb_ev`) | wOBA minus expected, fly balls | +13.4 | −7.5 to +34.8 | wOBA points |
  | H3a (`fb_rate`) | Home runs per fly ball | −0.46 | −1.64 to +0.90 | pp |
  | H3a (`fb_rate`) | Fly-ball distance | −0.5 | −2.7 to +1.8 | ft |
  | H3a (`fb_rate`) | wOBA minus expected, fly balls | −2.0 | −21.1 to +17.8 | wOBA points |
  | H2 (`ld_rate`) | In-park hit rate | −0.17 | −1.23 to +0.89 | pp |
  | H2 (`ld_rate`) | Home runs per PA | −0.19 | −0.42 to +0.04 | pp |
  | H1 (`brk_deficit`) | Run value, breaking balls | −0.12 | −0.41 to +0.15 | runs per 100 pitches |
  | H1 (`brk_deficit`) | Run value, fastballs | −0.30 | −0.49 to −0.09 | runs per 100 pitches |

- **Verdicts under D-30:**
  - **H1:** contradicts the hypothesis. The observed negative effect runs through fastball outcomes rather than breaking balls, so it is not the flattened-break mechanism.
  - **H2 and H3a:** not supported.
  - **H3b:** flagged as **likely noise**. It passed the confirmatory test, but no fly-ball outcome shows the carry mechanism.
  - **No pre-registered trait counts as a finding.**
- **Power caveat.** There are 4,419 Coors fly balls across 1,538 hitter-seasons. Interval widths are about ±1.6 pp for home runs per fly ball, ±3 ft for distance and ±21 points for wOBA over expected. Absence of mechanism evidence is not evidence of absence.
- **Not interpreted: the quality coefficients in these regressions.** They are strongly negative, but partly mechanical. "Elsewhere" is the raw same-season mean, and `quality` comes from the same non-Coors data, so luck elsewhere moves both. Trait coefficients are not affected, because traits come from earlier seasons.

## Phase 6: robustness (2026-10-21)

### D-32 · A trait with no detectable spread standardizes to zero
- **Decision:** if shrinkage finds no real spread in a trait for a season, so that every hitter shrinks to the same value, the standardized trait is 0 for everyone that season instead of undefined.
- **Why:** this happened once, for `brk_deficit` in 2025 within the stricter 400-PA population. The registered 200-PA population never hits it; earlier reports were unchanged byte for byte after the change. A unit test covers it.

### D-33 · Bootstrap seed sensitivity of the lift intervals
- **Finding:** the percentile endpoints of the lift intervals move by about ±15 (wOBA points)² depending on the bootstrap seed, because a few batters dominate the squared-error differences. Over ten seeds, the lower bound for unseen hitters ranged from −43 to −30 and the upper bound from 336 to 360. All of those intervals include zero.
- **Decision:** Phase 6 uses the same bootstrap stream as Phase 4, so the registered row reproduces Phase 4 exactly.

### D-34 · Phase 6 results
Full tables are in `reports/tables/phase6_robustness.md`.

| Variant | Hitter-seasons | H1 `brk_deficit` | H3b `fb_ev` | Lift, unseen hitters | Lift, future seasons |
|---|---:|---|---|---|---|
| Registered | 1,877 | −9.6, Holm p 0.008 | +12.3, Holm p 0.040 | +154 (−48 to +352) | −12 (−329 to +301) |
| Without 2020 | 1,819 | −9.3, Holm p 0.012 | +11.6, Holm p 0.074 (fails Holm) | +128 (−54 to +304) | −44 (−391 to +299) |
| Trait minimum 400 PAs | 1,536 | −9.9, Holm p 0.028 | +14.9, Holm p 0.024 | +44 (−94 to +185) | −57 (−415 to +282) |

- **The prediction null holds.** Every headline lift interval includes zero.
- **H1's reversed sign is stable,** at −9.3 to −9.9 and surviving Holm in every variant. Phase 5 found it does not run through breaking balls.
- **H3b's size is stable (+11.6 to +14.9), but its significance is fragile.** It fails Holm without 2020, which fits the Phase 5 "likely noise" flag.
- **No conclusion changes.**

## Phase 7: write-up (2026-10-21)

### D-35 · The memo and README results are generated, and their claims are checked
- **Decision:** `coorsfit.report` writes `reports/memo.md` and the README block between the `results` markers, using only results the pipeline saves. Phases 3 to 6 now also save their results as parquet for this purpose.
  - Statements the memo makes in words are asserted against the data before anything is written: intervals that include zero, H3b's verdict, H1's fastball pattern, and so on. If a rerun changes a result, `make report` fails instead of publishing stale wording.
  - The memo's prose template is exempt from the line-length lint rule.
- **The illustrative appendix:**
  - **Model:** M1 is fit on all 2016–2026 visitor hitter-seasons. Its trait coefficients are applied to 2027 traits built from 2025–2026 non-Coors data, for hitters with at least 200 such PAs. There is no same-season requirement, since 2027 hasn't been played.
  - **What is shown:** trait effects only. Quality is held at average, because it needs same-season data.
  - **Names:** from the MLB Stats API, cached in `data/raw/people.parquet`.
  - **Labeling:** it is labeled illustrative, and its spread is set against the gate's estimates.
