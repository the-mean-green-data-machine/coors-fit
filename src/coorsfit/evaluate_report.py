"""Phase 4 tables and figure, written from the results `coorsfit.evaluate` computes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coorsfit.clean import to_markdown  # noqa: E402
from coorsfit.config import ROOT, data_path  # noqa: E402
from coorsfit.targets import power  # noqa: E402
from coorsfit.targets_report import COLORS, INK, INK_2, MUTED, SURFACE, style  # noqa: E402

POINTS2 = 1e6  # wOBA^2 to (wOBA points)^2
SCHEME_LABEL = {
    "grouped_cv": "Unseen hitters (grouped CV)",
    "rolling_origin": "Future seasons (rolling origin)",
}


def lift_table(lifts: pd.DataFrame, ceiling: float) -> pd.DataFrame:
    t = lifts.copy()
    t["comparison"] = t["model"] + " vs " + t["baseline"]
    t["share_of_ceiling"] = (t["lift"] / ceiling).map("{:+.2f}".format)
    for c in ("lift", "lift_lo", "lift_hi"):
        t[c] = (t[c] * POINTS2).map("{:+.1f}".format)
    t["rmse_ratio"] = t["rmse_ratio"].map("{:.5f}".format)
    cols = ["scheme", "comparison", "role", "hitters", "coors_pas", "lift", "lift_lo", "lift_hi",
            "rmse_ratio", "share_of_ceiling"]  # fmt: skip
    return t[cols]


def coefficient_table(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["hypothesis", "term", "predicted_sign", "coef_points", "lo_points", "hi_points", "p",
            "p_holm", "sign_matches", "survives_holm"]  # fmt: skip
    out = res[cols].copy()
    out["hypothesis"] = out["hypothesis"].fillna("")
    out["predicted_sign"] = out["predicted_sign"].map({1.0: "+", -1.0: "-"}).fillna("")
    for c in ("sign_matches", "survives_holm"):
        out[c] = np.where(res["predicted_sign"].notna(), out[c].astype(str), "")
    out["p_holm"] = out["p_holm"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return out


def figure(lifts: pd.DataFrame, primary: pd.DataFrame, preds: pd.DataFrame, ceiling: float,
           mde: float) -> plt.Figure:  # fmt: skip
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.25, 1.1, 0.9]})  # fmt: skip

    style(a, "Out-of-sample lift per PA", "")
    a.grid(axis="y", visible=False)
    a.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    a.axvline(0, color=MUTED, linewidth=1)
    a.axvspan(0, ceiling * POINTS2, color="#eeede9", linewidth=0, zorder=0)
    labels, ticks, y = [], [], 0.0
    x_left = min(lifts["lift_lo"].min() * POINTS2, 0) * 1.05
    for scheme, color in (("grouped_cv", COLORS["visitor"]), ("rolling_origin", COLORS["rockies"])):
        # A group header names the validation scheme, so color is never the only cue.
        a.text(x_left, y + 0.15, SCHEME_LABEL[scheme], fontsize=8.5, color=color, va="bottom",
               fontweight="bold")  # fmt: skip
        y -= 0.75
        for _, r in lifts[lifts["scheme"] == scheme].iterrows():
            a.plot([r["lift_lo"] * POINTS2, r["lift_hi"] * POINTS2], [y, y], color=color,
                   linewidth=2.5, solid_capstyle="round")  # fmt: skip
            a.plot(r["lift"] * POINTS2, y, "o", color=color, markersize=6)
            labels.append(f"{r['model']} vs {r['baseline']}")
            ticks.append(y)
            y -= 1
        y -= 0.5
    a.set_yticks(ticks, labels, fontsize=8.5, color=INK)
    a.set_xlim(left=x_left)
    a.set_ylim(y + 0.6, 0.9)
    a.text(ceiling * POINTS2 * 0.97, 0.55, "ceiling", fontsize=8, color=INK_2, ha="right")
    a.set_xlabel("Squared-error reduction, (wOBA points)² per PA; shaded: up to the ceiling",
                 fontsize=8.5, color=INK_2)  # fmt: skip
    a.spines["bottom"].set_visible(True)

    style(b, "Confirmatory trait effects", "")
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    b.axvline(0, color=MUTED, linewidth=1)
    for x in (-mde, mde):
        b.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    conf = primary[primary["predicted_sign"].notna()].reset_index(drop=True)
    yy = np.arange(len(conf))[::-1]
    for yi, (_, r) in zip(yy, conf.iterrows(), strict=True):
        b.plot([r["lo_points"], r["hi_points"]], [yi, yi], color=COLORS["visitor"], linewidth=2.5,
               solid_capstyle="round")  # fmt: skip
        b.plot(r["coef_points"], yi, "o", color=COLORS["visitor"], markersize=6)
    tick_labels = [
        f"{h} {t} (predicted {'+' if s > 0 else '−'})"
        for h, t, s in zip(conf["hypothesis"], conf["term"], conf["predicted_sign"], strict=True)
    ]
    b.set_yticks(yy, tick_labels, fontsize=8.5, color=INK)
    b.set_xlabel(f"wOBA points per trait SD; dashed: ±{mde:.1f} minimum detectable", fontsize=8.5,
                 color=INK_2)  # fmt: skip
    b.spines["bottom"].set_visible(True)

    style(c, "Calibration of M1 (unseen hitters)", "Observed Coors residual (wOBA)")
    g = preds[preds["scheme"] == "grouped_cv"].copy()
    g["bin"] = pd.qcut(g["pred_M1"], 5, labels=False, duplicates="drop")
    rows = []
    for _, d in g.groupby("bin"):
        w = d["n"]
        obs = np.average(d["y"], weights=w)
        se = np.sqrt(np.sum(w**2 * d["noise_var"]) / w.sum() ** 2)
        rows.append((np.average(d["pred_M1"], weights=w), obs, se))
    xs, ys, ses = map(np.array, zip(*rows, strict=True))
    lo, hi = min(xs.min(), (ys - 2 * ses).min()), max(xs.max(), (ys + 2 * ses).max())
    c.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    c.errorbar(xs, ys, yerr=1.96 * ses, fmt="o", color=COLORS["visitor"], markersize=6,
               elinewidth=2, capsize=0)  # fmt: skip
    c.set_xlabel("M1 prediction (quintile means; dashed = perfect)", fontsize=8.5, color=INK_2)
    fig.tight_layout(w_pad=2.5)
    return fig


def write_report(visitors, preds, lifts, cover, conf, transfer, fits) -> None:
    gate = pd.read_parquet(data_path("processed") / "gate.parquet").iloc[0]
    ceiling = max(gate["tau2_hi"], 0.0)
    mde = 1000 * power(int(visitors["n"].sum()))["mde_80_alpha05"]
    primary = conf["primary (A-3: with quality)"]
    lt = lift_table(lifts, ceiling)

    fig = figure(lifts, primary, preds, ceiling, mde)
    fig.savefig(ROOT / "reports" / "figures" / "phase4_results.png", dpi=160, facecolor=SURFACE)

    head = lifts[(lifts["model"] == "M1") & (lifts["baseline"] == "B0q")].set_index("scheme")
    lines = []
    for scheme in ("grouped_cv", "rolling_origin"):
        r = head.loc[scheme]
        lines.append(
            f"- **{SCHEME_LABEL[scheme]}:** M1 vs B0q lift {r['lift'] * POINTS2:+.1f} "
            f"[{r['lift_lo'] * POINTS2:+.1f}, {r['lift_hi'] * POINTS2:+.1f}] "
            "(wOBA points)² per PA; "
            f"RMSE ratio {r['rmse_ratio']:.5f}."
        )
    found = primary[primary["survives_holm"] & primary["sign_matches"]]
    lines.append(
        f"- **Confirmatory tests:** {len(found)} of 5 traits have the predicted sign and survive "
        "the Holm correction"
        + (f" ({', '.join(found['hypothesis'] + ' ' + found['term'])})." if len(found) else ".")
        + " A trait counts as a finding only if its Phase 5 mechanism check also agrees."
    )
    lines.append(
        f"- **Rockies transfer:** calibration slope {transfer['slope']:+.2f} "
        f"[{transfer['slope_lo']:+.2f}, {transfer['slope_hi']:+.2f}] across "
        f"{transfer['rockies_hitters']} hitters."
    )

    sections = [
        "# Phase 4 results",
        "Written by `make models` (`coorsfit.evaluate`). "
        "Figure: `reports/figures/phase4_results.png`.",
        "Lift is the reduction in squared error per held-out visiting Coors PA against the "
        "baseline, in (wOBA points)²; positive means the model is better. The ceiling is the "
        f"gate's upper bound on τ², {ceiling * POINTS2:.0f} (wOBA points)² per PA: no forecast of "
        "hitters' true Coors effects could do better. Intervals resample batters (95%).",
        "## Summary",
        "\n".join(lines),
        "## Out-of-sample lift",
        to_markdown(lt),
        "## Confirmatory tests (primary, amendment A-3)",
        f"PA-weighted least squares on {len(visitors):,} eligible visitor hitter-seasons "
        f"({int(visitors['n'].sum()):,} Coors PAs). Coefficients are wOBA points per trait SD "
        "(quality: per unit of expected wOBA, ×1000); the intercept is the league Coors effect. "
        "Holm correction across the five confirmatory traits.",
        to_markdown(coefficient_table(primary)),
        "## Sensitivity: without the quality control",
        to_markdown(coefficient_table(conf["sensitivity: no quality"])),
        "## Exploratory: adding sprint speed (H5)",
        to_markdown(coefficient_table(conf["exploratory: with sprint speed (H5)"])),
        "## Predictive interval coverage (held out)",
        "Intervals are the prediction ± the hitter-season's noise SD (per-PA variance over its "
        "Coors PAs plus expectation error) at the 50% and 90% levels.",
        to_markdown(cover),
        "## Rockies transfer test",
        "M1 fit on all visitors predicts Rockies hitters' hitter-season residuals. Both "
        "predictions and outcomes are centered on the Rockies average without the hitter's own "
        "rows; the slope regresses outcome on prediction, weighted by Coors PAs. Lift is per PA "
        "against that Rockies average, in (wOBA points)².",
        to_markdown(
            pd.DataFrame([transfer]).assign(
                lift_vs_rockies_mean=lambda d: (d["lift_vs_rockies_mean"] * POINTS2).map(
                    "{:+.1f}".format
                )
            )
        ),
        "## Ridge penalties chosen inside training",
        to_markdown(
            fits.groupby(["scheme", "model"])["alpha"]
            .describe()[["min", "50%", "max"]]
            .reset_index()
        ),
    ]
    out = ROOT / "reports" / "tables" / "phase4_results.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(out.read_text())
