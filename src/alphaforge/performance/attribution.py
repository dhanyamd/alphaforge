"""Return attribution — where did the P&L actually come from?

Using the same factor decomposition as the risk model, split each period's
portfolio return into:
  * **factor return**  = portfolio factor exposure · factor return  (the bets you
    intended — value, momentum, sectors, ...),
  * **specific return** = the rest (name-selection skill / idiosyncratic noise).

This is how you tell skill from luck and find *where* the skill lives, so you
double down on what works and cut what doesn't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attribute_period(
    weights: np.ndarray,
    exposures: np.ndarray,          # (N, K) portfolio names' factor exposures
    factor_return: np.ndarray,      # (K,) realized factor returns this period
    asset_return: np.ndarray,       # (N,) realized asset returns this period
    factor_names: list[str],
) -> dict[str, float]:
    """Attribute one period's portfolio return to factors vs specific."""
    port_factor_expo = weights @ exposures              # (K,) portfolio's factor bet
    factor_contrib = port_factor_expo * factor_return   # (K,) per-factor P&L
    total_return = float(weights @ asset_return)
    factor_total = float(factor_contrib.sum())
    specific_total = total_return - factor_total
    out = {"total": total_return, "factor_total": factor_total, "specific_total": specific_total}
    for name, c in zip(factor_names, factor_contrib, strict=False):
        out[f"f_{name}"] = float(c)
    return out


def attribution_table(period_attrs: list[dict[str, float]]) -> pd.DataFrame:
    """Aggregate per-period attributions into a cumulative table (sum of contributions)."""
    df = pd.DataFrame(period_attrs)
    summary = df.sum(numeric_only=True).to_frame("cumulative_contribution")
    summary["share_of_total"] = summary["cumulative_contribution"] / (
        summary.loc["total", "cumulative_contribution"] or np.nan
    )
    return summary
