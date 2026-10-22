"""Phase 5 tests: the Coors-minus-elsewhere metric and the D-30 verdict rules."""

import pandas as pd

from coorsfit.mechanisms import coors_minus_elsewhere, verdicts


def test_coors_minus_elsewhere_and_its_weight():
    events = pd.DataFrame(
        {
            "batter": 1,
            "season": 2020,
            "is_coors": [True] * 4 + [False] * 12,
            "value": [1.0, 1.0, 0.0, 0.0] + [0.0] * 9 + [1.0] * 3,
        }
    )
    rows = pd.DataFrame({"batter": [1, 2], "season": 2020})
    out = coors_minus_elsewhere(events, rows)
    assert len(out) == 1  # batter 2 has no events and is dropped
    assert out["y"].iloc[0] == 0.5 - 0.25
    assert out["n"].iloc[0] == 4 * 12 / 16


def checks_frame(coefs: dict[tuple[str, str], tuple[float, float, float]]) -> pd.DataFrame:
    own = {"rv100_breaking": "brk_deficit", "rv100_fastball": "brk_deficit",
           "inpark_ld_fb_hit_rate": "ld_rate", "hr_per_pa": "ld_rate", "hr_per_fb": "fb_ev",
           "fb_distance": "fb_ev", "fb_woba_minus_x": "fb_ev"}  # fmt: skip
    hyp = {m: {"brk_deficit": "H1", "ld_rate": "H2", "fb_ev": "H3"}[t] for m, t in own.items()}
    rows = []
    for metric, trait in own.items():
        for term in ("brk_deficit", "ld_rate", "fb_rate", "fb_ev", "k_rate"):
            coef, lo, hi = coefs.get((metric, term), (0.0, -1.0, 1.0))
            rows.append({"metric": metric, "hypothesis": hyp[metric], "term": term, "coef": coef,
                         "lo": lo, "hi": hi, "own_trait": term == trait})  # fmt: skip
    return pd.DataFrame(rows)


CONF = pd.DataFrame(
    {
        "term": ["brk_deficit", "ld_rate", "fb_rate", "fb_ev", "k_rate"],
        "coef_points": [-9.0, 1.0, 1.0, 12.0, -1.0],
        "p_holm": [0.01, 0.9, 0.9, 0.04, 0.9],
        "survives_holm": [True, False, False, True, False],
        "sign_matches": [False, True, True, True, True],
    }
)


def test_verdicts_follow_the_rules():
    v = verdicts(checks_frame({}), CONF).set_index("hypothesis")["verdict"]
    assert v["H1"].startswith("contradicts")
    assert v["H2"].startswith("not supported")
    assert v["H3b"].startswith("likely noise")  # passed, but no fly-ball outcome is significant

    agreeing = checks_frame({("fb_distance", "fb_ev"): (3.0, 1.0, 5.0)})
    assert (
        verdicts(agreeing, CONF).set_index("hypothesis").loc["H3b", "verdict"].startswith("finding")
    )

    conflicting = checks_frame(
        {("fb_distance", "fb_ev"): (3.0, 1.0, 5.0), ("hr_per_fb", "fb_ev"): (-2.0, -3.0, -1.0)}
    )
    assert (
        verdicts(conflicting, CONF)
        .set_index("hypothesis")
        .loc["H3b", "verdict"]
        .startswith("likely noise")
    )
