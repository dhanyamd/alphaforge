"""Factor covariance + specific variance estimation (EWMA + shrinkage).

Two estimates feed the risk decomposition V = B Σ_f B' + D:

  * **Σ_f (factor covariance)** — EWMA of factor returns (recent days weighted
    more, via a half-life), then **shrunk** toward a diagonal target to tame the
    sampling error that always plagues a covariance estimated from finite data.

  * **D (specific variances)** — per name, EWMA of squared specific returns,
    with Bayesian **shrinkage toward the cross-sectional median** so a name with
    little history doesn't get an absurd variance.

Half-lives and shrinkage intensity come from config (typical: 90d cov, 60d var,
shrink ~0.1-0.3 — consistent with how Barra-style models are run in production).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Weights for n observations (oldest..newest) with the given half-life."""
    lam = np.log(2) / halflife
    ages = np.arange(n)[::-1]            # newest has age 0
    w = np.exp(-lam * ages)
    return w / w.sum()


def factor_covariance(
    factor_returns: pd.DataFrame,
    halflife: float = 90,
    shrinkage: float = 0.2,
) -> pd.DataFrame:
    """EWMA factor covariance with diagonal shrinkage. Returns a K x K frame.

    Shrinkage: Σ = (1-δ)·Σ_sample + δ·diag(Σ_sample). Pulls noisy off-diagonals
    toward zero — a simple, robust Ledoit-Wolf-style regularization.
    """
    F = factor_returns.dropna(how="all").fillna(0.0)
    n, k = F.shape
    if n < 2:
        return pd.DataFrame(np.eye(k) * 1e-6, index=F.columns, columns=F.columns)
    w = _ewma_weights(n, halflife)
    X = F.values
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc           # weighted covariance
    cov = cov / (1 - (w**2).sum())           # bias correction for weights
    target = np.diag(np.diag(cov))
    cov = (1 - shrinkage) * cov + shrinkage * target
    return pd.DataFrame(cov, index=F.columns, columns=F.columns)


def specific_variance(
    specific_returns: pd.DataFrame,
    halflife: float = 60,
    shrinkage: float = 0.2,
    floor: float = 1e-6,
) -> pd.Series:
    """Per-name specific variance (EWMA of squared specific return) + shrinkage.

    Returns a Series indexed by security_id. Shrinks each name's estimate toward
    the cross-sectional median to stabilize thin-history names.
    """
    sr = specific_returns.copy()
    sr["date"] = pd.to_datetime(sr["date"])
    sr = sr.sort_values(["security_id", "date"])

    def _ewvar(s: pd.Series) -> float:
        x = s.dropna().values
        if len(x) == 0:
            return np.nan
        w = _ewma_weights(len(x), halflife)
        return float(np.average(x**2, weights=w))

    var = sr.groupby("security_id")["specific"].apply(_ewvar)
    med = np.nanmedian(var.values)
    var = (1 - shrinkage) * var + shrinkage * med
    return var.clip(lower=floor)
