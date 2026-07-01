"""Cross-sectional regression — estimate factor returns from realized returns.

This is the Barra-style estimation. Each day we already KNOW every name's factor
*exposures* B_{t-1} (sector membership + style z-scores, lagged so they're
point-in-time). We regress the cross-section of realized returns on those
exposures:

        r_{i,t} = Σ_k B_{i,k,t-1} f_{k,t} + u_{i,t}

The fitted coefficients f_{k,t} are the **factor returns** for day t (how much
each factor "paid" that day); the residuals u_{i,t} are the **specific returns**.
We run this every day to build a time series of factor returns (-> factor
covariance) and specific returns (-> specific risk). We weight by sqrt(market
cap) — bigger, more reliable names anchor the fit (standard Barra practice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_exposures(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    style_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Construct the exposure matrix columns: sector dummies + style z-scores.

    Returns (exposures_long, factor_names). Exposures are already standardized
    (the styles are z-scores from the alpha layer); sector dummies are 0/1.
    """
    sec_map = static.set_index("security_id")["sector"]
    df = panel.copy()
    df["sector"] = df["security_id"].map(sec_map)
    sector_dummies = pd.get_dummies(df["sector"], prefix="SEC").astype(float)
    factor_names = list(sector_dummies.columns) + list(style_cols)
    expo = pd.concat([df[["date", "security_id"]], sector_dummies, df[style_cols]], axis=1)
    return expo, factor_names


def estimate_factor_returns(
    expo: pd.DataFrame,
    returns: pd.DataFrame,
    factor_names: list[str],
    weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the daily cross-sectional WLS regressions.

    Parameters
    ----------
    expo : long frame with [date, security_id, <factor_names>] — exposures at t-1.
    returns : long frame [date, security_id, fwd_ret] — the return realized over
              the period the exposures are predicting (already aligned/lagged).
    weights : optional [date, security_id, weight] regression weights (sqrt cap).

    Returns
    -------
    factor_returns : wide [date x factor_names]
    specific_returns : long [date, security_id, specific]
    """
    merged = expo.merge(returns, on=["date", "security_id"], how="inner")
    if weights is not None:
        merged = merged.merge(weights, on=["date", "security_id"], how="left")
        merged["weight"] = merged["weight"].fillna(merged["weight"].median())
    else:
        merged["weight"] = 1.0

    fac_rows: list[dict] = []
    spec_rows: list[pd.DataFrame] = []
    for date, cross in merged.groupby("date"):
        X = cross[factor_names].values.astype(float)
        y = cross["fwd_ret"].values.astype(float)
        w = np.sqrt(cross["weight"].values.astype(float))
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < len(factor_names) + 5:
            continue
        Xw = X[ok] * w[ok, None]
        yw = y[ok] * w[ok]
        # WLS via least squares on weighted system; pinv-safe for collinear dummies.
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y[ok] - X[ok] @ beta
        fac_rows.append({"date": date, **dict(zip(factor_names, beta, strict=False))})
        spec_rows.append(pd.DataFrame({
            "date": date,
            "security_id": cross["security_id"].values[ok],
            "specific": resid,
        }))

    factor_returns = pd.DataFrame(fac_rows).set_index("date").sort_index()
    specific_returns = (
        pd.concat(spec_rows, ignore_index=True) if spec_rows else
        pd.DataFrame(columns=["date", "security_id", "specific"])
    )
    return factor_returns, specific_returns
