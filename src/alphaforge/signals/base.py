"""Cross-sectional signal hygiene — the transforms every signal goes through.

Raw signals are messy: fat tails, different scales, and contaminated by things
you don't want to bet on (sector, size). Before a signal is usable you:

  1. **winsorize** — clip extreme outliers so one bad print doesn't dominate.
  2. **standardize (z-score)** — put every signal on the same cross-sectional
     scale (mean 0, std 1) so they can be combined.
  3. **neutralize** — regress out exposures you don't want (e.g. sector, size),
     keeping only the *residual* view. This is what makes a signal a clean,
     independent bet instead of a disguised sector tilt.

These run *per date* (one cross-section at a time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def neutralize(s: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Regress `s` on `factors` (incl. dummies) and return the residual.

    OLS residual = the part of the signal orthogonal to the unwanted exposures.
    Dropping nuisance exposures (sector, size) is what turns a raw signal into a
    genuinely independent bet — central to the "breadth" idea in the fundamental
    law of active management.
    """
    # Sanitize: real vendor data has NaN/inf (missing fundamentals, failed
    # downloads). Feeding those to lstsq throws "SVD did not converge". Replace
    # non-finite values with 0 so the regression is always well-posed.
    y = np.nan_to_num(s.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.nan_to_num(factors.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.column_stack([np.ones(len(y)), X])  # intercept
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except np.linalg.LinAlgError:
        resid = y - y.mean()   # degenerate cross-section -> just demean
    return pd.Series(resid, index=s.index)


def clean_signal(
    raw: pd.Series,
    sector: pd.Series | None = None,
    size: pd.Series | None = None,
) -> pd.Series:
    """Full hygiene pipeline for a single cross-section: winsorize -> neutralize -> z."""
    # Coerce to numeric and kill non-finite values up front (real data is messy).
    s = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.median())
    if s.isna().all():
        s = s.fillna(0.0)
    s = winsorize(s)
    if sector is not None or size is not None:
        cols = {}
        if sector is not None:
            dummies = pd.get_dummies(sector, drop_first=True).astype(float)
            for c in dummies.columns:
                cols[f"sec_{c}"] = dummies[c].values
        if size is not None:
            sz = pd.to_numeric(size, errors="coerce").replace([np.inf, -np.inf], np.nan)
            cols["size"] = zscore(sz.fillna(sz.median())).values
        if cols:
            s = neutralize(s, pd.DataFrame(cols, index=s.index))
    return zscore(s)
