"""Phase 5 mechanism checks for H1 to H3 (rules fixed in DECISIONS.md, D-30, before running).

For each eligible visitor hitter-season, a mechanism outcome at Coors (his visiting PAs only)
minus the same outcome at every other park that season, regressed on the confirmatory
specification (five traits plus quality) with a batter bootstrap.

Writes reports/tables/phase5_mechanisms.md and reports/figures/phase5_mechanisms.png.
Run with `python -m coorsfit.mechanisms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coorsfit.clean import to_markdown  # noqa: E402
from coorsfit.config import ROOT, data_path, load_config  # noqa: E402
from coorsfit.evaluate import coefficient_tests, confirmatory, modeling_table  # noqa: E402
from coorsfit.features import CONFIRMATORY  # noqa: E402
from coorsfit.targets_report import COLORS, INK, INK_2, MUTED, SURFACE, style  # noqa: E402


@dataclass
class Metric:
    name: str
    hypothesis: str
    trait: str
    unit: str
    scale: float  # multiplies the raw value into `unit`
    role: str  # "channel": should carry the effect; "control": should not


METRICS = [
    Metric("rv100_breaking", "H1", "brk_deficit", "runs per 100 pitches", 100, "channel"),
    Metric("rv100_fastball", "H1", "brk_deficit", "runs per 100 pitches", 100, "comparison"),
    Metric("inpark_ld_fb_hit_rate", "H2", "ld_rate", "percentage points", 100, "channel"),
    Metric("hr_per_pa", "H2", "ld_rate", "percentage points", 100, "control"),
    Metric("hr_per_fb", "H3", "fb_ev", "percentage points", 100, "channel"),
    Metric("fb_distance", "H3", "fb_ev", "feet", 1, "channel"),
    Metric("fb_woba_minus_x", "H3", "fb_ev", "wOBA points", 1000, "channel"),
]


def event_values() -> dict[str, pd.DataFrame]:
    """Per-event values for each metric: batter, season, at_coors, value."""
    P = data_path("processed")
    cols = ["batter", "season", "is_coors", "home_player", "bb_type", "is_hr", "is_hit",
            "hit_distance_sc", "woba_value", "xwoba_bip", "is_woba_pa"]  # fmt: skip
    pa = pd.read_parquet(P / "pa.parquet", columns=cols)
    pa = pa[~(pa["is_coors"] & pa["home_player"])]  # Coors means visiting PAs only
    fb = pa["bb_type"] == "fly_ball"
    out = {
        "hr_per_fb": pa[fb].assign(value=pa["is_hr"].astype(float)),
        "fb_distance": pa[fb & pa["hit_distance_sc"].notna()].assign(value=pa["hit_distance_sc"]),
        "fb_woba_minus_x": pa[fb & pa["is_woba_pa"] & pa["xwoba_bip"].notna()].assign(
            value=pa["woba_value"] - pa["xwoba_bip"]
        ),
        "inpark_ld_fb_hit_rate": pa[
            pa["bb_type"].isin(["line_drive", "fly_ball"]) & ~pa["is_hr"]
        ].assign(value=pa["is_hit"].astype(float)),
        "hr_per_pa": pa[pa["is_woba_pa"]].assign(value=pa["is_hr"].astype(float)),
    }
    groups = load_config()["traits"]["pitch_groups"]
    first, last = load_config()["seasons"]["targets"]
    pcols = ["batter", "season", "is_coors", "home_player", "pitch_type", "delta_run_exp"]
    pitches = pd.concat(
        pd.read_parquet(data_path("interim") / f"pitches_{s}.parquet", columns=pcols)
        for s in range(first, last + 1)
    )
    pitches = pitches[
        pitches["delta_run_exp"].notna() & ~(pitches["is_coors"] & pitches["home_player"])
    ]
    for name, key in (("rv100_breaking", "breaking"), ("rv100_fastball", "fastball")):
        p = pitches[pitches["pitch_type"].isin(groups[key])]
        out[name] = p.assign(value=p["delta_run_exp"])
    return {k: v[["batter", "season", "is_coors", "value"]] for k, v in out.items()}


def coors_minus_elsewhere(events: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per eligible hitter-season: mean at Coors minus mean elsewhere, inverse-variance weighted."""
    agg = events.groupby(["batter", "season", "is_coors"])["value"].agg(["sum", "size"]).unstack()
    agg.columns = [f"{s}_{'coors' if c else 'other'}" for s, c in agg.columns]
    df = rows.merge(agg.reset_index(), on=["batter", "season"], how="inner")
    df = df[(df["size_coors"] > 0) & (df["size_other"] > 0)].copy()
    nc, no = df["size_coors"].astype(float), df["size_other"].astype(float)
    df["y"] = df["sum_coors"] / nc - df["sum_other"] / no
    df["n"] = nc * no / (nc + no)
    df["coors_events"] = nc
    return df


def run_checks(rows: pd.DataFrame) -> pd.DataFrame:
    values = event_values()
    results = []
    for i, m in enumerate(METRICS):
        df = coors_minus_elsewhere(values[m.name], rows)
        res = coefficient_tests(df, [*CONFIRMATORY, "quality"], stream=500 + i)
        for col in ("coef_points", "lo_points", "hi_points", "se_points"):
            res[col] = res[col] / 1000 * m.scale  # coefficient_tests reports ×1000
        res = res.rename(columns={"coef_points": "coef", "lo_points": "lo", "hi_points": "hi",
                                  "se_points": "se"})  # fmt: skip
        avg = np.average(df["y"], weights=df["n"]) * m.scale
        for _, r in res.iterrows():
            results.append(
                {
                    "metric": m.name,
                    "hypothesis": m.hypothesis,
                    "role": m.role,
                    "unit": m.unit,
                    "hitter_seasons": len(df),
                    "coors_events": int(df["coors_events"].sum()),
                    "coors_minus_elsewhere": avg,
                    "term": r["term"],
                    "coef": r["coef"],
                    "lo": r["lo"],
                    "hi": r["hi"],
                    "p": r["p"],
                    "own_trait": r["term"] == m.trait,
                }  # fmt: skip
            )
    return pd.DataFrame(results)


def verdicts(checks: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """Apply the D-30 rules."""
    own = checks[checks["own_trait"]].set_index("metric")
    sig_pos = lambda m: own.loc[m, "lo"] > 0  # noqa: E731
    c = conf.set_index("term")
    h3 = ["hr_per_fb", "fb_distance", "fb_woba_minus_x"]
    rules = {
        "H1": ("brk_deficit", sig_pos("rv100_breaking")
               and own.loc["rv100_breaking", "coef"] > own.loc["rv100_fastball", "coef"]),
        "H2": ("ld_rate", sig_pos("inpark_ld_fb_hit_rate") and not sig_pos("hr_per_pa")),
    }  # fmt: skip
    # H3a and H3b share the fly-ball outcomes; each is checked with its own trait's coefficients.
    for h, trait in (("H3a", "fb_rate"), ("H3b", "fb_ev")):
        coefs = checks[(checks["hypothesis"] == "H3") & (checks["term"] == trait)].set_index(
            "metric"
        )
        agrees = (coefs.loc[h3, "lo"] > 0).any() and not (coefs.loc[h3, "hi"] < 0).any()
        rules[h] = (trait, bool(agrees))
    out = []
    for h, (trait, agrees) in rules.items():
        passed = bool(c.loc[trait, "survives_holm"] and c.loc[trait, "sign_matches"])
        wrong_sign = bool(c.loc[trait, "survives_holm"] and not c.loc[trait, "sign_matches"])
        if passed and agrees:
            verdict = "finding: passes the confirmatory test and its mechanism agrees"
        elif passed:
            verdict = "likely noise: passes the confirmatory test without mechanism evidence"
        elif wrong_sign:
            verdict = "contradicts the hypothesis (significant, opposite sign)"
        else:
            verdict = "not supported (confirmatory test not passed)"
        out.append(
            {
                "hypothesis": h,
                "trait": trait,
                "confirmatory_coef": c.loc[trait, "coef_points"],
                "confirmatory_p_holm": c.loc[trait, "p_holm"],
                "mechanism_agrees": bool(agrees),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out)


def h1_direction(checks: pd.DataFrame) -> str:
    own = checks[checks["own_trait"]].set_index("metric")
    b, f = own.loc["rv100_breaking"], own.loc["rv100_fastball"]
    through_breaking = b["hi"] < 0 and b["coef"] < f["coef"]
    return (
        f"`brk_deficit` on breaking-ball run value: {b['coef']:+.2f} "
        f"[{b['lo']:+.2f}, {b['hi']:+.2f}]; on fastball run value: {f['coef']:+.2f} "
        f"[{f['lo']:+.2f}, {f['hi']:+.2f}] (runs per 100 pitches per SD). "
        "The observed negative H1 effect "
        + (
            "**does** run through breaking balls."
            if through_breaking
            else "does **not** clearly run through breaking balls."
        )
    )


def figure(checks: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), facecolor=SURFACE)
    titles = {
        "rv100_breaking": "H1 · Run value vs breaking balls",
        "rv100_fastball": "H1 · Run value vs fastballs",
        "inpark_ld_fb_hit_rate": "H2 · Hit rate, in-park LD+FB",
        "hr_per_pa": "H2 · HR per PA (control)",
        "hr_per_fb": "H3 · HR per fly ball",
        "fb_distance": "H3 · Fly-ball distance",
        "fb_woba_minus_x": "H3 · wOBA − xwOBA on fly balls",
    }
    labels = {"brk_deficit": "H1 brk_deficit", "ld_rate": "H2 ld_rate", "fb_rate": "H3a fb_rate",
              "fb_ev": "H3b fb_ev", "k_rate": "H4 k_rate"}  # fmt: skip
    for ax, m in zip(axes.flat, METRICS, strict=False):
        d = checks[(checks["metric"] == m.name) & checks["term"].isin(CONFIRMATORY)].reset_index(
            drop=True
        )
        style(ax, titles[m.name], "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=1)
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows(), strict=True):
            color = (
                COLORS["visitor"]
                if r["own_trait"] or (m.hypothesis == "H3" and r["term"] == "fb_rate")
                else "#b4b2ac"
            )
            ax.plot(
                [r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.2, solid_capstyle="round"
            )
            ax.plot(r["coef"], yi, "o", color=color, markersize=5)
        ax.set_yticks(y, [labels[t] for t in d["term"]], fontsize=8, color=INK)
        ax.set_xlabel(f"{m.unit} per trait SD", fontsize=8, color=INK_2)
        ax.spines["bottom"].set_visible(True)
    last = axes.flat[-1]
    last.axis("off")
    note = (
        "Each panel: Coors minus elsewhere for one\noutcome, regressed on all five traits\n"
        "plus quality; 95% batter-bootstrap intervals.\n\n"
        "Blue: the trait(s) the hypothesis names.\nGray: the other traits, for contrast."
    )
    last.text(0, 0.9, note, fontsize=8.5, color=INK_2, va="top", transform=last.transAxes)
    fig.tight_layout(h_pad=2.2, w_pad=1.8)
    return fig


def main() -> None:
    visitors = modeling_table("visitor")
    rows = visitors[["batter", "season", "quality", *CONFIRMATORY]]
    checks = run_checks(rows)
    conf = confirmatory(visitors)["primary (A-3: with quality)"]
    v = verdicts(checks, conf)
    checks.to_parquet(data_path("processed") / "phase5_checks.parquet", index=False)
    v.to_parquet(data_path("processed") / "phase5_verdicts.parquet", index=False)

    fig = figure(checks)
    fig.savefig(ROOT / "reports" / "figures" / "phase5_mechanisms.png", dpi=160, facecolor=SURFACE)

    own = checks[
        checks["own_trait"] | ((checks["hypothesis"] == "H3") & (checks["term"] == "fb_rate"))
    ]
    own_cols = ["hypothesis", "metric", "role", "term", "unit", "hitter_seasons", "coors_events",
                "coors_minus_elsewhere", "coef", "lo", "hi", "p"]  # fmt: skip
    sections = [
        "# Phase 5 mechanism checks",
        "Written by `make mechanisms` (`coorsfit.mechanisms`). Rules fixed before running "
        "(DECISIONS.md, D-30). Figure: `reports/figures/phase5_mechanisms.png`.",
        "Each outcome is the hitter-season's mean at Coors (visiting PAs) minus his mean at other "
        "parks that season, regressed on the five confirmatory traits plus quality, weighted by "
        "inverse variance, with 95% batter-bootstrap intervals. `coors_minus_elsewhere` is the "
        "weighted average difference (the Coors effect on that outcome). Coefficients are per SD "
        "of the trait, in the outcome's unit.",
        "## Verdicts",
        to_markdown(v),
        "## H1: does the observed negative effect run through breaking balls?",
        h1_direction(checks),
        "## Each hypothesis's own trait",
        to_markdown(own[own_cols]),
        "## All coefficients",
        to_markdown(
            checks[checks["term"] != "intercept"][["metric", "term", "coef", "lo", "hi", "p"]]
        ),
    ]
    out = ROOT / "reports" / "tables" / "phase5_mechanisms.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
