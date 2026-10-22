"""Phase 4 models: B0, B0q, M1, M1u and M2. Every fit sees training rows only.

Rows are eligible hitter-seasons. The outcome `y` is the mean Coors residual and `n` its Coors
PAs. Weighted least squares on these rows with weights `n` is the same fit as least squares on
the PAs, because a hitter-season's features are constant across its PAs.

- B0: the league-average Coors effect (an intercept).
- B0q: B0 plus a slope on `quality` (amendment A-3).
- M1: B0q plus the traits, ridge-penalized; intercept and quality are not penalized, so any
  gain over B0q comes from the traits. The penalty is picked by grouped CV inside training.
- M1u: M1 without `quality` (sensitivity, compared with B0).
- M2: a small LightGBM on M1's features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from coorsfit.config import load_config, rng
from coorsfit.features import CONFIRMATORY

TRAIT_FEATURES = [*CONFIRMATORY, "sprint_speed"]


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, penalty: np.ndarray | None = None):
    """Weighted least squares with an optional diagonal ridge penalty per coefficient."""
    XtW = X.T * w
    A = XtW @ X
    if penalty is not None:
        A = A + np.diag(penalty)
    return np.linalg.solve(A, XtW @ y)


def design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), *(df[c].to_numpy(dtype=float) for c in cols)])


def batter_folds(batters: pd.Series, k: int, stream: int) -> np.ndarray:
    """Assign each batter (and so all his rows) to one of k folds at random."""
    ids = batters.unique()
    fold_of = pd.Series(rng(stream).permutation(len(ids)) % k, index=ids)
    return batters.map(fold_of).to_numpy()


@dataclass
class Linear:
    """B0, B0q, M1 and M1u: an intercept, unpenalized controls, and ridge-penalized traits."""

    name: str
    controls: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    alpha: float = 0.0
    beta: np.ndarray | None = None

    @property
    def cols(self) -> list[str]:
        return [*self.controls, *self.traits]

    def _penalty(self, alpha: float) -> np.ndarray:
        return np.array([0.0] * (1 + len(self.controls)) + [alpha] * len(self.traits))

    def fit(self, train: pd.DataFrame) -> Linear:
        X, y, w = design(train, self.cols), train["y"].to_numpy(), train["n"].to_numpy(float)
        if self.traits:
            self.alpha = self._choose_alpha(train)
        self.beta = wls(X, y, w, self._penalty(self.alpha))
        return self

    def _choose_alpha(self, train: pd.DataFrame) -> float:
        cfg = load_config()["models"]
        folds = batter_folds(train["batter"], cfg["inner_folds"], stream=101)
        X, y, w = design(train, self.cols), train["y"].to_numpy(), train["n"].to_numpy(float)
        errors = []
        for alpha in cfg["ridge_alphas"]:
            sse = 0.0
            for f in range(cfg["inner_folds"]):
                tr, te = folds != f, folds == f
                b = wls(X[tr], y[tr], w[tr], self._penalty(alpha))
                sse += np.sum(w[te] * (y[te] - X[te] @ b) ** 2)
            errors.append(sse)
        return float(cfg["ridge_alphas"][int(np.argmin(errors))])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return design(test, self.cols) @ self.beta


@dataclass
class Boosted:
    """M2: a small LightGBM on the same features as M1."""

    name: str
    cols: list[str]
    model: object = None

    def fit(self, train: pd.DataFrame) -> Boosted:
        from lightgbm import LGBMRegressor

        params = load_config()["models"]["lgbm"]
        self.model = LGBMRegressor(
            **params, random_state=load_config()["seed"], deterministic=True,
            force_row_wise=True, n_jobs=1, verbose=-1,
        )  # fmt: skip
        self.model.fit(train[self.cols], train["y"], sample_weight=train["n"])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test[self.cols])


def model_set() -> list:
    return [
        Linear("B0"),
        Linear("B0q", controls=["quality"]),
        Linear("M1", controls=["quality"], traits=TRAIT_FEATURES),
        Linear("M1u", traits=TRAIT_FEATURES),
        Boosted("M2", cols=[*TRAIT_FEATURES, "quality"]),
    ]
