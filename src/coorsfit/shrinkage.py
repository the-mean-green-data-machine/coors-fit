"""Empirical-Bayes helpers: estimate how much groups truly differ, then shrink their means."""

from __future__ import annotations

import numpy as np


def tau2_anova(
    sums: np.ndarray,
    counts: np.ndarray,
    sigma2: float,
    extra_var: np.ndarray | None = None,
    clip: bool = True,
) -> float:
    """Between-group variance by method of moments (one-way ANOVA, unbalanced groups).

    `sums` and `counts` are each group's total and number of observations; `sigma2` is the
    variance of one observation within a group. `extra_var` is any further known variance of
    each group's mean that is not true difference, such as error in the expectation the
    residuals are measured from. With `clip`, returns 0 when the spread between group means is
    no more than that noise would produce; without it, the raw (possibly negative) estimate.
    """
    n = np.asarray(counts, dtype=float)
    s = np.asarray(sums, dtype=float)
    v = np.zeros_like(n) if extra_var is None else np.asarray(extra_var, dtype=float)
    keep = n > 0
    n, s, v = n[keep], s[keep], v[keep]
    total = n.sum()
    grand = s.sum() / total
    between = np.sum(n * (s / n - grand) ** 2)
    noise = np.sum((1 - n / total) * (sigma2 + n * v))
    estimate = float((between - noise) / (total - np.sum(n**2) / total))
    return max(0.0, estimate) if clip else estimate


def design_effect(
    values: np.ndarray, groups: np.ndarray, clusters: np.ndarray, sigma2: float
) -> float:
    """How much clustering inflates the sampling variance of each group's mean (at least 1).

    Observations in one cluster (a group's PAs in one game) can share a shock, such as a
    pitcher's bad day. Compares the spread of cluster totals around each group's mean with
    what independent observations of variance `sigma2` would give.
    """
    groups = np.asarray(groups, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    width = clusters.max() + 1
    cells, cell_of = np.unique(groups * width + clusters, return_inverse=True)
    cell_sum = np.bincount(cell_of, weights=values)
    cell_n = np.bincount(cell_of).astype(float)
    group_of_cell = cells // width
    group_n = np.bincount(group_of_cell, weights=cell_n)
    group_mean = np.bincount(group_of_cell, weights=cell_sum) / group_n
    spread = np.sum((cell_sum - cell_n * group_mean[group_of_cell]) ** 2)
    expected = sigma2 * np.sum(group_n - np.bincount(group_of_cell, weights=cell_n**2) / group_n)
    return max(1.0, float(spread / expected))


def shrink(
    sums: np.ndarray,
    counts: np.ndarray,
    sigma2: float,
    tau2: float,
    extra_var: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Posterior mean of each group's effect under a Normal(0, tau2) prior.

    A group's observed mean has variance sigma2 / n plus `extra_var` around its true effect.
    """
    sums = np.asarray(sums, dtype=float)
    if tau2 <= 0:
        return np.zeros_like(sums)
    n = np.asarray(counts, dtype=float)
    return (sums / n) * tau2 / (tau2 + extra_var + sigma2 / n)


def cluster_slope(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x with a standard error clustered on `clusters` (e.g. batter)."""
    X = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    _, codes = np.unique(clusters, return_inverse=True)
    scores = np.column_stack([np.bincount(codes, weights=X[:, j] * u) for j in range(2)])
    cov = xtx_inv @ (scores.T @ scores) @ xtx_inv
    return float(beta[1]), float(np.sqrt(cov[1, 1]))
