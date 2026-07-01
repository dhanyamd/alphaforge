"""Predictive-power analysis — does the model's alpha actually forecast returns?

This is the evidence that the strategy has a real edge (as opposed to the unit
tests, which only prove the plumbing is correct). Two classic diagnostics every
quant runs on a signal:

  * **IC (information coefficient)** — the cross-sectional rank correlation
    between the model's prediction (alpha) and the return that actually came
    next. Positive = it predicts. Real single-name equity ICs are tiny (0.02-
    0.05); a small but persistently-positive IC is the whole game.

  * **Quantile spread** — each period, sort names into quintiles by alpha and
    measure the forward return of each bucket. A *monotonic* staircase (the names
    it liked beat the names it disliked) is the signature of a working factor
    model, and the single most convincing chart in a quant interview.

Everything here is point-in-time: alpha at date D is compared to the return
realized AFTER D, never before.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictiveReport:
    mean_ic: float
    ic_ir: float                 # IC's own information ratio (mean/std * sqrt(252))
    pct_days_positive: float
    quantile_returns: pd.Series  # index 1..5, avg forward return per bucket
    top_minus_bottom_bps: float  # daily Q5-Q1 spread in basis points
    n_obs: int


def _daily_ic(group: pd.DataFrame, pred_col: str, fwd_col: str) -> float:
    if group[pred_col].std(ddof=0) == 0 or len(group) < 10:
        return np.nan
    return group[pred_col].rank().corr(group[fwd_col].rank())


def analyze_predictive_power(
    alpha_panel: pd.DataFrame,
    prices: pd.DataFrame,
    pred_col: str = "alpha",
    quantiles: int = 5,
) -> PredictiveReport:
    """Compute IC + quantile spread of `pred_col` against next-day returns.

    Parameters
    ----------
    alpha_panel : long frame with [date, security_id, <pred_col>].
    prices : long frame with [date, security_id, ret]; we derive the forward
             (next-day) return internally so the comparison is strictly causal.
    """
    px = prices.sort_values(["security_id", "date"]).copy()
    px["date"] = pd.to_datetime(px["date"])
    px["fwd_ret"] = px.groupby("security_id")["ret"].shift(-1)   # tomorrow's return

    df = alpha_panel[["date", "security_id", pred_col]].merge(
        px[["date", "security_id", "fwd_ret"]], on=["date", "security_id"]
    ).dropna()

    # --- IC time series ---
    ic = df.groupby("date").apply(lambda g: _daily_ic(g, pred_col, "fwd_ret")).dropna()
    mean_ic = float(ic.mean())
    ic_ir = float(ic.mean() / ic.std(ddof=0) * np.sqrt(252)) if ic.std(ddof=0) > 0 else 0.0
    pct_pos = float((ic > 0).mean())

    # --- quantile forward returns ---
    def _q(group: pd.DataFrame) -> pd.Series:
        if len(group) < quantiles * 2:
            return pd.Series(dtype=float)
        g = group.copy()
        g["bucket"] = pd.qcut(g[pred_col].rank(method="first"), quantiles,
                              labels=list(range(1, quantiles + 1)))
        return g.groupby("bucket", observed=True)["fwd_ret"].mean()

    qret = df.groupby("date").apply(_q).mean()
    if isinstance(qret, pd.DataFrame):  # flatten if needed
        qret = qret.mean()
    top_bottom_bps = float((qret.iloc[-1] - qret.iloc[0]) * 1e4)

    return PredictiveReport(
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        pct_days_positive=pct_pos,
        quantile_returns=qret,
        top_minus_bottom_bps=top_bottom_bps,
        n_obs=len(df),
    )
