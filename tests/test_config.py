"""Phase 0 checks: the settings every stage relies on are consistent, and the environment loads."""

from coorsfit.config import ROOT, load_config, rng


def test_season_ranges_line_up():
    seasons = load_config()["seasons"]
    first, last = seasons["pull"]
    t_first, t_last = seasons["targets"]
    # Every target season needs at least one earlier pulled season to build traits from.
    assert first < t_first <= t_last == last
    assert t_first < load_config()["validation"]["rolling_origin_first_test"] <= t_last
    assert all(t_first <= s <= t_last for s in seasons["shortened"])


def test_thresholds_are_ordered():
    traits = load_config()["traits"]
    assert 0 < traits["min_prior_noncoors_pa"] < traits["sensitivity_min_prior_noncoors_pa"]
    assert load_config()["target"]["min_same_season_noncoors_pa"] > 0


def test_pitch_groups_do_not_overlap():
    groups = load_config()["traits"]["pitch_groups"]
    assert not set(groups["fastball"]) & set(groups["breaking"])


def test_rng_streams_are_reproducible_and_distinct():
    assert (rng(1).random(5) == rng(1).random(5)).all()
    assert not (rng(1).random(5) == rng(2).random(5)).all()


def test_preregistration_lists_every_hypothesis():
    text = (ROOT / "docs" / "hypotheses.md").read_text()
    for h in ("H1", "H2", "H3a", "H3b", "H4", "H5"):
        assert f"| {h}" in text


def test_lightgbm_loads():
    # Fails when the OpenMP runtime is missing; `make setup` installs it.
    import lightgbm

    assert lightgbm.__version__
