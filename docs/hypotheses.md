# Pre-registered hypotheses and analysis plan

Committed on 2026-10-21, before any model was fit and before any trait was compared with any Coors outcome. The only data looked at beforehand were environment checks: field availability, missing-data rates, counts of Coors PAs per hitter, overall visitor and Rockies-hitter wOBA at Coors and on the road in 2025, and breaking-ball movement at Coors versus elsewhere (see DECISIONS.md, D-04).

Anything changed after this commit goes under **Amendments** at the bottom, with a date and reason, and must be logged before the step it affects is run. Any analysis not described here is reported in a section labeled exploratory.

## Question

Given only a hitter's non-Coors history, can we forecast how much more or less than the league-average Coors effect he gains at Coors Field?

## Data rules

- **Coors** means a regular-season game at Coors Field (MLB Stats API venue 19). `home_team == 'COL'` alone is not enough: the Rockies were the home team for two 2024 games in Mexico City.
- **Primary sample:** visiting hitters at Coors, target seasons 2016 to 2026. The headline result and all five confirmatory tests use this sample only. Rockies hitters at home are a pre-registered secondary analysis, always reported separately (see below).
- **Coors data appears only in the target.** Features use non-Coors PAs from seasons before the target season. Same-season non-Coors PAs may set the expectation the residual is measured against, but are never a feature.
- **The league-average Coors effect is fit inside each training fold**, never on held-out players or seasons.
- **Eligibility:** at least 200 non-Coors PAs across the two seasons before the target season, and at least 100 non-Coors PAs in the target season. There is no minimum number of Coors PAs.

## Target

- **Outcome per PA:** Statcast `woba_value`, on PAs with `woba_denom == 1`, in additive wOBA points.
- **Hitter-season Coors residual:** the mean, over a hitter's visiting Coors PAs, of the PA's `woba_value` minus its expected value. The expected value comes from the hitter's same-season non-Coors level, the pitcher's same-season non-Coors level, platoon side and season, plus the league-average Coors effect estimated on training data.
- **Regressions use the raw residual, weighted by Coors PAs.** This is equivalent to a PA-level model with trait-by-Coors interactions and the expectation as an offset. Empirical-Bayes shrinkage is applied for per-hitter estimates, the gate, and displays, not to the regression target, because shrinking the target would bias the coefficients.

## Hypotheses and traits

Every trait is computed from non-Coors PAs in the two seasons before the target season (one season for 2016 targets, since 2015 is the first season pulled). Each is shrunk toward the league mean by empirical Bayes, then standardized within the target season among eligible hitters, so every coefficient is in wOBA points of Coors residual per 1 SD of the trait. Player identity is never a feature.

| ID | Trait (non-Coors, prior seasons) | Predicted sign | Mechanism | Mechanism check |
|---|---|---|---|---|
| H1 | `brk_deficit`: the hitter's run value per 100 pitches against fastballs (FF, SI, FC) minus against breaking balls (SL, ST, SV, CU, KC, CS) | + | Breaking balls lose movement at altitude (in 2025 about a quarter less, both directions), so a hitter's weakness against them shrinks. | The effect shows up in the hitter's Coors outcomes on breaking balls more than on fastballs. |
| H2 | `ld_rate`: line drives per batted ball | + | Coors has the largest outfield in MLB, so more line drives and shallow flies fall in. The infield is regulation size, so ground balls are not expected to benefit. | The effect shows up in hit rate on in-park line drives and fly balls, not in home-run rate. |
| H3a | `fb_rate`: fly balls per batted ball (pop-ups excluded) | + | Thin air adds carry to fly balls. | The effect shows up in home runs per fly ball, in fly-ball distance, and in actual minus expected wOBA on fly balls (expected wOBA ignores the park). |
| H3b | `fb_ev`: mean exit velocity on fly balls | + | Same carry mechanism as H3a. | Same checks as H3a. |
| H4 | `k_rate`: strikeouts per PA | − | In additive wOBA points, the Coors boost works through balls in play, so hitters who put fewer balls in play gain less. | The effect largely disappears when the Coors residual is measured per batted ball rather than per PA. |
| H5 (exploratory) | `sprint_speed`: Savant sprint speed, most recent prior season | + | Speed matters more on a large outfield. | The effect shows up in hits on balls in play and extra bases. |

**H4 is the positive control.** It is close to mechanical, so if the design cannot detect H4, it lacks the power to judge H1 to H3, and nulls on those mean "can't tell" rather than "no effect."

**H5 is exploratory and is not counted in the confirmatory tests.** The Savant sprint-speed leaderboard includes runs at Coors and cannot be restricted to non-Coors PAs. This is a documented exception to the non-Coors rule: it is a physical measurement, taken from prior seasons only.

## Confirmatory tests

- **Five tests:** the coefficients for H1, H2, H3a, H3b and H4 in one PA-weighted least-squares fit containing all five traits.
- **Intervals:** 2,000 bootstrap replicates resampled by batter.
- **Multiple comparisons:** Holm correction at a family-wise α of 0.05. All five results are reported, whatever they show.
- **A trait counts as a finding** only if its sign matches the prediction, it survives the Holm correction, and its mechanism check agrees. A significant trait with no mechanism evidence is flagged as likely noise. A significant effect with the wrong sign is reported as contradicting its hypothesis.

## Gate: is there a stable Coors skill? (Phase 3)

- **Primary:** τ, the between-hitter standard deviation of the true Coors effect, pooled across all eligible seasons. It is estimated by method of moments (the PA-weighted variance of hitter-level residuals minus their mean sampling variance), with a bootstrap interval by batter. If the lower bound is above zero, that is evidence of a stable skill. Otherwise we report the upper bound as "hitters differ by less than X." τ is estimated for visitors (the gate) and, separately, for Rockies hitters.
- **Secondary:** the correlation between a hitter's odd-season and even-season Coors residuals, over his career, for hitters with at least 30 visiting Coors PAs in each half, with a bootstrap interval.
- **Phase 4 runs whatever the gate shows.** We will not switch to a different target to rescue a null.

## Prediction and scoring (Phase 4)

- **B0** gives every hitter the league-average Coors effect, estimated on the training fold.
- **M1** is the five confirmatory traits plus `sprint_speed` in a PA-weighted ridge regression. The penalty is chosen by inner cross-validation grouped by batter.
- **M2** is a small LightGBM model on the same features, as a nonlinear check.
- **Primary metric:** the mean per-PA reduction in squared error relative to B0 on held-out visiting Coors PAs, scored against raw outcomes, with a 95% interval from a bootstrap that resamples batters. We also report the RMSE ratio.
- **Ceiling:** a perfect forecast of each hitter's true effect would cut per-PA squared error by τ². Lift is reported next to this ceiling.
- **Validation schemes:**
  - Five-fold cross-validation grouped by batter, over 2016 to 2026, so every test hitter is unseen.
  - Rolling origin: for each test season Y from 2021 to 2026, train on target seasons before Y.
- **Calibration:** coverage of 50% and 90% predictive intervals for hitter-season Coors residuals, plus a calibration plot.

## Secondary analysis: Rockies hitters at home

This analysis is pre-registered, but it is not part of the headline or the confirmatory tests, and it is always reported separately from visitors.

**Why it is secondary.** A Rockies hitter's non-Coors data are road games, which altitude may itself distort. In 2025, Rockies hitters had a .268 wOBA on the road against .318 for road hitters league-wide; one season cannot separate talent from a "Coors hangover." A dozen Rockies regulars per season would also carry about half of all Coors PAs. H1 is the most exposed: road struggles against breaking balls would raise both `brk_deficit` and the Coors residual, which are measured against the same road games, so pooling could confirm H1 for the wrong reason.

**Sample.** Rockies hitters' PAs at Coors, under the same eligibility thresholds. Their traits and same-season expectation use only non-Coors PAs (road games and seasons with other teams), so their trait estimates are thinner.

**Fields.**
- `home_player`: batting for the Rockies at Coors.
- `tenure_seasons`: the number of earlier seasons, counted from 2015, in which the hitter had a PA for the Rockies. This is left-censored for hitters who joined before 2015.

**Transfer test (the main Rockies result).**
- Fit M1 on visitors only and predict Rockies hitters.
- Score with the calibration slope: regress Rockies hitter-season Coors residuals on the visitor model's predicted deviations, weighted by Coors PAs. Both are centered on the Rockies average, estimated leaving each player out in turn. The 95% interval comes from resampling batters.
- A slope near 1 means visitor effects carry over to home players. A slope near 0 means they do not, which is itself a finding about adaptation.
- We also report per-PA lift over a B0 that includes the Rockies average.

**Pooled model.**
- The M1 specification is fit on visitors and Rockies hitters together, with `home_player` and `tenure_seasons` as covariates.
- Its B0 includes separate league-average Coors effects for visitors and for Rockies hitters, both fit on the training fold, so lift measures the traits rather than the group shift.
- Evaluated with held-out-player CV and rolling origin, and reported by group.

**Reading.** Tenure is read descriptively only, because hitters who do well at Coors are more likely to stay.

**Not in the minimum version:**
- Trait-by-`home_player` interactions (exploratory if time allows).
- The movers extension (the first thing to cut).

## Power (planning estimate)

From the 2025 sizing (per-PA wOBA SD 0.514, about 3,300 visiting Coors PAs per full season), 2016 to 2026 should hold about 35,000 visiting Coors PAs, of which we assume about 25,000 survive eligibility. That gives each trait coefficient a standard error of about 0.003 wOBA per trait SD. The minimum detectable effect at 80% power is then about **9 wOBA points per trait SD** (two-sided α of 0.05), or about **11 points** at the Bonferroni-equivalent α of 0.01.

In Phase 2, this estimate is recomputed from the actual eligible PA count (counts only, no outcomes) and logged as an amendment if it changes. A null result is reported as "effects larger than the minimum detectable effect are unlikely."

## Robustness (Phase 6, minimum scope)

- Exclude 2020.
- Raise the trait minimum to 400 prior non-Coors PAs.

None of these is confirmatory.

## Amendments

### A-1 · 2026-10-21 · The expectation model gains three adjustments

Logged in Phase 2, before any trait was computed or compared with a Coors outcome. The target is unchanged: Coors wOBA minus the expected wOBA from same-season non-Coors data. Only the expectation model changes.

- **A home/road term.** A visitor's non-Coors PAs include his home games, where every team hits a little better. Without this term, his expected level at Coors, where he is the road team, would be set too high.
- **Park effects for non-Coors parks, shrunk toward zero.** A Coors PA is predicted at a park effect of zero, so the residual measures Coors against the average non-Coors park rather than against the hitter's own home park.
- **Pitchers' own PAs are left out of the fit** (before 2022, about 5,000 per season). Their very low wOBA made regular hitters look more different from each other than they are, which led to too little shrinkage.
- **Shrinkage accounts for PAs within a game sharing shocks,** such as a pitcher's bad day. It does this through a design effect estimated from game-to-game variation.

Evidence: DECISIONS.md, D-19 to D-21, and `reports/tables/targets_diagnostics.md`.

### A-2 · 2026-10-21 · Power confirmed from eligible counts

There are 25,275 eligible visitor Coors PAs in 2016 to 2026. With the pre-registered per-PA SD of 0.514, the minimum detectable effect at 80% power is 9.1 wOBA points per trait SD at α = 0.05 and 11.1 at α = 0.01. This matches the planning estimate, so nothing else changes.

### A-3 · 2026-10-21 · Control for hitter quality in the confirmatory fit and the models

Logged after Phase 2 found that visitors' Coors residuals slope against their expected wOBA (−0.28 ± 0.12; DECISIONS.md, D-23), and before any trait was computed. Several traits track hitter quality, so without a control that slope could leak into their coefficients.

- **The control, `quality`:** a hitter-season's mean expected wOBA over its Coors PAs, minus the PA-weighted mean of the same quantity among eligible hitter-seasons in its group and season.
  - It comes from the same-season non-Coors expectation that already defines the residual, so it refines the expectation rather than adding a trait.
  - It is not a hypothesis and is not standardized with the traits.
- **Confirmatory fit:** the five confirmatory coefficients come from the PA-weighted least-squares fit with `quality` added as a control. The Holm correction and the bootstrap are unchanged.
- **Models:** M1 includes `quality` alongside the traits. M2 uses the same features as M1, so it includes `quality` too.
- **Baselines:**
  - The headline comparison is M1 against **B0q**: the league-average Coors effect plus a slope on `quality`, both fit on the training fold. Any lift over B0q therefore comes from the traits, not from quality.
  - The pre-registered B0 (league-average effect only) is reported as well, which shows how much the quality slope adds on its own.
- **Sensitivity:** the uncontrolled versions (the confirmatory fit and M1 without `quality`, scored against B0) are reported as a sensitivity analysis.
- **Gate:** unchanged. τ is estimated as registered; τ net of the quality slope is reported alongside it as supplementary.

### A-4 · 2026-10-21 · Two refinements to the gate's noise term

These were adopted while building Phase 3. The statistic (τ), the decision rule (lower bound above zero) and the interval (bootstrap over hitters) are as registered. Only the noise subtracted in the method of moments changes.

- **Remove the expectation's own error.**
  - A hitter's residuals are measured against his estimated batter level. That estimate's error (posterior SD about 0.019 for visitors) is shared by all his Coors PAs that season, so it would otherwise count as skill.
  - Its variance, which is independent across seasons, is subtracted.
  - This was designed before the gate was run, and it can only lower τ.
- **Use each hitter's own per-PA variance.** Hitters with more Coors PAs tend to have more volatile outcomes, so a single pooled variance overstates the noise.
  - This was adopted after Phase 2's hitter-season estimate for visitors came out implausibly negative (τ² = −0.0028), and before the hitter-level gate was run.
  - It raises τ, so it works against a null.

Evidence: DECISIONS.md, D-25. Unit tests check that the estimator recovers a known spread and stays near zero on pure noise.
