"""Phase 7: the front-office memo and the README's results block, written from saved outputs.

Every number in reports/memo.md and between the README's results markers comes from files the
pipeline writes, so `make report` keeps both in step with the analysis.

Run with `python -m coorsfit.report`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from coorsfit.clean import to_markdown
from coorsfit.config import ROOT, data_path, load_config
from coorsfit.evaluate import modeling_table
from coorsfit.features import build_features, load_inputs
from coorsfit.ingest import fetch_names
from coorsfit.models import TRAIT_FEATURES, model_set
from coorsfit.targets import power

P2 = 1e6  # wOBA^2 to (wOBA points)^2
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"
APPENDIX_ROWS = 8


def load() -> dict:
    P = data_path("processed")
    r = {name: pd.read_parquet(P / f"{name}.parquet") for name in (
        "gate", "gate_split", "phase4_lifts", "phase4_confirmatory", "phase4_transfer",
        "phase5_checks", "phase5_verdicts", "phase6_coefs", "phase6_lifts",
        "shrinkage_diagnostics", "targets", "expectation_fits",
    )}  # fmt: skip
    pa = pd.read_parquet(P / "pa.parquet", columns=["season", "is_woba_pa", "woba_value"])
    first, last = load_config()["seasons"]["targets"]
    in_range = pa["is_woba_pa"] & pa["season"].between(first, last)
    r["league_woba"] = float(pa.loc[in_range, "woba_value"].mean())
    return r


def illustrative_forecasts(visitors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trait-only M1 predictions for the season after the data ends (labelled illustrative)."""
    cfg = load_config()
    m1 = next(m for m in model_set() if m.name == "M1").fit(visitors)
    trait_beta = pd.Series(m1.beta[1 + len(m1.controls) :], index=m1.traits)

    pa, pitches = load_inputs()
    target = cfg["seasons"]["pull"][1] + 1
    window = range(target - cfg["traits"]["window_seasons"], target)
    prior = pa[pa["is_woba_pa"] & ~pa["is_coors"] & pa["season"].isin(window)]
    counts = prior.groupby("batter").size()
    population = pd.DataFrame(
        {"batter": counts[counts >= cfg["traits"]["min_prior_noncoors_pa"]].index, "season": target}
    )
    feats = build_features(pa, pitches, population)
    feats["pred_points"] = 1000 * feats[TRAIT_FEATURES].to_numpy() @ trait_beta.to_numpy()
    ranked = feats.sort_values("pred_points", ascending=False)
    shown = pd.concat([ranked.head(APPENDIX_ROWS), ranked.tail(APPENDIX_ROWS)])
    names = fetch_names(shown["batter"].astype(int).tolist())
    table = pd.DataFrame(
        {
            "hitter": shown["batter"].astype(int).map(names).fillna("(unknown)").to_numpy(),
            "predicted_vs_average_points": shown["pred_points"].map("{:+.1f}".format).to_numpy(),
            "fly_ball_ev_mph": shown["fb_ev_raw"].map("{:.1f}".format).to_numpy(),
            "breaking_ball_deficit": shown["brk_deficit_raw"].map("{:+.2f}".format).to_numpy(),
            "sprint_speed": shown["sprint_speed_raw"].map("{:.1f}".format).to_numpy(),
        }
    )
    meta = {
        "season": target,
        "window": f"{window.start}–{window.stop - 1}",
        "hitters": len(feats),
        "sd_points": float(feats["pred_points"].std()),
        "alpha": m1.alpha,
        "beta_points": (1000 * trait_beta).round(1).to_dict(),
    }
    return table, meta


def numbers(r: dict, visitors: pd.DataFrame) -> dict:
    """Every figure the memo and README quote, in the units they quote it."""
    conf = r["phase4_confirmatory"]
    primary = conf[conf["spec"] == "primary (A-3: with quality)"].set_index("term")
    explore = conf[conf["spec"] == "exploratory: with sprint speed (H5)"].set_index("term")
    lifts = r["phase4_lifts"].set_index(["scheme", "model", "baseline"])
    head_cv = lifts.loc[("grouped_cv", "M1", "B0q")]
    head_ro = lifts.loc[("rolling_origin", "M1", "B0q")]
    gate = r["gate"].iloc[0]
    split = r["gate_split"].iloc[0]
    tr = r["phase4_transfer"].iloc[0]
    checks = r["phase5_checks"]
    own = checks[checks["term"] == "fb_ev"].set_index("metric")
    brk = checks[checks["term"] == "brk_deficit"].set_index("metric")
    shrink = r["shrinkage_diagnostics"].set_index("group").loc["visitor"]
    t = r["targets"]
    v = t[t["eligible"] & (t["group"] == "visitor")]
    p6 = r["phase6_coefs"].set_index(["variant", "term"])
    no2020 = p6.loc[("Without 2020", "fb_ev")]
    sigma_points = 1000 * shrink["sigma"]
    median_season_pas = float(v["n_coors"].median())
    amendments = len(re.findall(r"^### A-\d+", (ROOT / "docs/hypotheses.md").read_text(), re.M))
    p6_lifts = r["phase6_lifts"]
    return {
        "first": load_config()["seasons"]["targets"][0],
        "last": load_config()["seasons"]["targets"][1],
        "hitter_seasons": len(visitors),
        "hitters": visitors["batter"].nunique(),
        "coors_pas": int(visitors["n"].sum()),
        "effect": primary.loc["intercept", "coef_points"],
        "effect_lo": primary.loc["intercept", "lo_points"],
        "effect_hi": primary.loc["intercept", "hi_points"],
        "effect_pct": 100 * primary.loc["intercept", "coef_points"] / 1000 / r["league_woba"],
        "league_woba": r["league_woba"],
        "tau_hi": 1000 * gate["tau_hi"],
        "tau": 1000 * gate["tau"],
        "gate_hitters": int(gate["hitters"]),
        "gate_median_pas": gate["median_pas_per_hitter"],
        "split_r": split["r"], "split_lo": split["r_lo"], "split_hi": split["r_hi"],
        "split_n": int(split["hitters"]),
        "lift_cv": head_cv["lift"] * P2, "lift_cv_lo": head_cv["lift_lo"] * P2,
        "lift_cv_hi": head_cv["lift_hi"] * P2, "rmse_cv": head_cv["rmse_ratio"],
        "lift_ro": head_ro["lift"] * P2, "lift_ro_lo": head_ro["lift_lo"] * P2,
        "lift_ro_hi": head_ro["lift_hi"] * P2,
        "ceiling": max(gate["tau2_hi"], 0) * P2,
        "ceiling_pct": 100 * max(gate["tau2_hi"], 0) / shrink["sigma"] ** 2,
        "fbev": primary.loc["fb_ev", "coef_points"], "fbev_p": primary.loc["fb_ev", "p_holm"],
        "fbev_hrfb": own.loc["hr_per_fb", "coef"], "fbev_hrfb_lo": own.loc["hr_per_fb", "lo"],
        "fbev_hrfb_hi": own.loc["hr_per_fb", "hi"],
        "fbev_dist": own.loc["fb_distance", "coef"], "fbev_dist_lo": own.loc["fb_distance", "lo"],
        "fbev_dist_hi": own.loc["fb_distance", "hi"],
        "fbev_no2020_p": no2020["p_holm"],
        "brk": primary.loc["brk_deficit", "coef_points"], "brk_p": primary.loc["brk_deficit", "p_holm"],
        "brk_fast": brk.loc["rv100_fastball", "coef"], "brk_fast_lo": brk.loc["rv100_fastball", "lo"],
        "brk_fast_hi": brk.loc["rv100_fastball", "hi"],
        "brk_break": brk.loc["rv100_breaking", "coef"], "brk_break_lo": brk.loc["rv100_breaking", "lo"],
        "brk_break_hi": brk.loc["rv100_breaking", "hi"],
        "fb_distance_gain": own.loc["fb_distance", "coors_minus_elsewhere"],
        "hrfb_gain": own.loc["hr_per_fb", "coors_minus_elsewhere"],
        "sprint": explore.loc["sprint_speed", "coef_points"],
        "sprint_lo": explore.loc["sprint_speed", "lo_points"],
        "sprint_hi": explore.loc["sprint_speed", "hi_points"],
        "transfer": tr["slope"], "transfer_lo": tr["slope_lo"], "transfer_hi": tr["slope_hi"],
        "transfer_n": int(tr["rockies_hitters"]),
        "median_season_pas": median_season_pas,
        "noise_season": sigma_points / np.sqrt(median_season_pas),
        "calibration": r["expectation_fits"]["slope"].mean(),
        "mde": 1000 * power(int(visitors["n"].sum()))["mde_80_alpha05"],
        "amendments": amendments,
        "robust_lifts_include_zero": bool(((p6_lifts["lift_lo"] < 0) & (p6_lifts["lift_hi"] > 0)).all()),
    }  # fmt: skip


def memo(n: dict, appendix: pd.DataFrame, meta: dict) -> str:
    woba = f"{n['league_woba']:.3f}".lstrip("0")
    rmse_pct = 100 * (1 - n["rmse_cv"])
    return f"""# Coors Fit: can we tell in advance which hitters gain more at Coors?

*Memo for the front office. Data: Statcast, visiting hitters at Coors Field, {n["first"]}–{n["last"]}. Every number is generated by `make report` from the analysis outputs.*

## Bottom line

Visiting hitters do about **{n["effect"]:.0f} wOBA points better at Coors** than their non-Coors form and the pitchers they face predict (95% CI {n["effect_lo"]:.0f} to {n["effect_hi"]:.0f}), a boost of about {n["effect_pct"]:.0f}% on a league-average {woba} wOBA. **Beyond that average, we found no reliable way to tell in advance which hitters will gain more or less.**

## What we found

- **No detectable hitter-specific Coors skill.** True Coors effects differ between hitters by less than {n["tau_hi"]:.0f} wOBA points (SD, upper 95% bound; best estimate {n["tau"]:.0f}). A hitter's odd-season and even-season Coors results are uncorrelated (r = {n["split_r"]:+.2f}, 95% CI {n["split_lo"]:+.2f} to {n["split_hi"]:+.2f}).
- **Hitter traits did not improve forecasts.** Measured on hitters and seasons the model never saw, adding six traits to the baseline (the league-average effect with a hitter-quality adjustment) gained {n["lift_cv"]:+.0f} (95% CI {n["lift_cv_lo"]:+.0f} to {n["lift_cv_hi"]:+.0f}) in squared error for unseen hitters and {n["lift_ro"]:+.0f} ({n["lift_ro_lo"]:+.0f} to {n["lift_ro_hi"]:+.0f}) for future seasons, in (wOBA points)² per PA; negative means worse than the baseline. For unseen hitters that is a {rmse_pct:.2f}% change in prediction error.
- **The two trait signals did not hold up.**
  - Fly-ball exit velocity had the predicted effect (+{n["fbev"]:.1f} points per SD), but those hitters hit no extra home runs per fly ball ({n["fbev_hrfb"]:+.1f} pp) and gained no extra distance ({n["fbev_dist"]:+.1f} ft) at Coors, where fly balls carry {n["fb_distance_gain"]:.0f} ft farther. It also lost significance without 2020. Likely noise.
  - Hitters weaker against breaking balls gained *less* ({n["brk"]:+.1f} points per SD), the opposite of our prediction, and the gap showed up against fastballs, not breaking balls.
- **Rockies hitters are unresolved.** Visitor-based forecasts carried over with a slope of {n["transfer"]:+.2f} (95% CI {n["transfer_lo"]:+.2f} to {n["transfer_hi"]:+.2f}), from only {n["transfer_n"]} hitters.

## What to do with it

1. **Apply one league-average Coors adjustment to every visiting hitter.** Don't pay for, or discount, a hitter's supposed "Coors fit."
2. **Ignore individual Coors splits.** A visitor's season at Coors is about {n["median_season_pas"]:.0f} PAs, which carries roughly {n["noise_season"]:.0f} wOBA points of noise.
3. **Value acquisitions on park-neutral talent,** and expect the average boost to come with them. Whether everyday Rockies hitters adapt differently is still open.
4. **Test next, under a fresh pre-registration:** the reversed breaking-ball pattern, and sprint speed, which looked positive in an exploratory fit (+{n["sprint"]:.1f} points per SD, 95% CI {n["sprint_lo"]:+.1f} to {n["sprint_hi"]:+.1f}).

## How sure we are

Even a perfect forecast of each hitter's Coors effect could cut per-PA squared error by at most {n["ceiling"]:.0f} (wOBA points)², under {n["ceiling_pct"]:.1f}% of one PA's variance. So individual differences, if they exist, are small.

The design guards against false positives:
- Hypotheses were fixed in advance; {n["amendments"]} amendments were each logged before the step they changed.
- Traits use only earlier seasons away from Coors, and that is tested.
- Models are scored only on unseen hitters or seasons.
- Traits that passed had to show their mechanism.

Results hold without 2020 and with a 400-PA trait minimum. The limits:
- The primary analysis covers visitors only.
- There is no weather data.
- The expectation model spreads slightly too wide (held-out calibration slope {n["calibration"]:.2f}).
- The mechanism checks have limited power.

## Appendix: illustrative only, not a forecast to act on

What the trait model would predict for {meta["season"]}, from {meta["window"]} non-Coors traits, as each hitter's Coors effect relative to the average visitor (wOBA points). Across {meta["hitters"]} hitters the predictions spread with an SD of {meta["sd_points"]:.1f} points: wider than the best estimate of true differences ({n["tau"]:.0f}), close to its upper bound ({n["tau_hi"]:.0f}), and small next to the {n["noise_season"]:.0f}-point noise in one Coors season. The model did not beat the baseline out of sample. Top and bottom {APPENDIX_ROWS}:

{to_markdown(appendix)}
"""


def readme_block(n: dict) -> str:
    return f"""{README_START}
**Headline (generated by `make report`):**

- **The average Coors effect is real.** Visiting hitters do **+{n["effect"]:.0f} wOBA points** better
  at Coors than expected (95% CI {n["effect_lo"]:.0f} to {n["effect_hi"]:.0f}).
- **There is no detectable hitter-specific Coors skill.** True effects differ by an SD of less than
  {n["tau_hi"]:.0f} points (upper 95% bound; point estimate {n["tau"]:.0f}).
- **Traits don't beat a league-average baseline out of sample.** The lift is {n["lift_cv"]:+.0f}
  ({n["lift_cv_lo"]:+.0f} to {n["lift_cv_hi"]:+.0f}) for unseen hitters and {n["lift_ro"]:+.0f}
  ({n["lift_ro_lo"]:+.0f} to {n["lift_ro_hi"]:+.0f}) for future seasons, in (wOBA points)² per PA.
  The ceiling is {n["ceiling"]:.0f}.
- **No pre-registered trait counts as a finding.**
  - **Fly-ball exit velocity** (+{n["fbev"]:.1f} per SD) failed its mechanism check.
  - **The breaking-ball deficit** came out with the opposite sign ({n["brk"]:+.1f} per SD).
- **These results hold** without 2020 and with a 400-PA trait minimum.

Read the [memo](reports/memo.md) first.
{README_END}"""


def check_claims(n: dict, r: dict) -> None:
    """The memo states some results in words; fail loudly if a rerun makes any of them false."""
    verdict = r["phase5_verdicts"].set_index("hypothesis")["verdict"]
    claims = {
        "no stable skill (gate)": not bool(r["gate"].iloc[0]["stable_skill"]),
        "split-half interval includes zero": n["split_lo"] < 0 < n["split_hi"],
        "headline lift intervals include zero": n["lift_cv_lo"] < 0 < n["lift_cv_hi"]
        and n["lift_ro_lo"] < 0 < n["lift_ro_hi"],
        "robustness lifts include zero": n["robust_lifts_include_zero"],
        "H3b flagged likely noise": verdict["H3b"].startswith("likely noise"),
        "H3b fails Holm without 2020": n["fbev_no2020_p"] >= 0.05,
        "H1 opposite sign": n["brk"] < 0 and verdict["H1"].startswith("contradicts"),
        "H1 gap is in fastballs, not breaking balls": n["brk_fast_hi"] < 0 < n["brk_break_hi"],
        "fly-ball EV shows no HR/FB gain": n["fbev_hrfb_lo"] < 0 < n["fbev_hrfb_hi"],
        "fly-ball EV shows no distance gain": n["fbev_dist_lo"] < 0 < n["fbev_dist_hi"],
        "Rockies transfer interval includes zero": n["transfer_lo"] < 0 < n["transfer_hi"],
        "no pre-registered finding": not verdict.str.startswith("finding").any(),
    }
    failed = [k for k, ok in claims.items() if not ok]
    assert not failed, f"memo wording no longer matches the results: {failed}"


def main() -> None:
    r = load()
    visitors = modeling_table("visitor")
    appendix, meta = illustrative_forecasts(visitors)
    n = numbers(r, visitors)
    check_claims(n, r)
    (ROOT / "reports" / "memo.md").write_text(memo(n, appendix, meta))
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + readme_block(n) + text[end:])
    print((ROOT / "reports" / "memo.md").read_text())
    print("M1 trait coefficients (points per SD):", meta["beta_points"], "alpha", meta["alpha"])


if __name__ == "__main__":
    main()
