"""Implementation shortfall — the honest measure of what trading cost you.

Run two portfolios in parallel:
  * a **paper** portfolio that rebalances with ZERO trading costs and instant
    full fills (the idealized strategy), and
  * the **real** portfolio with commission, spread, impact, and participation caps.

The gap between their cumulative returns IS the implementation shortfall — the
total value the act of trading subtracted. Decomposed, it tells you whether the
leak is spread, impact, or opportunity cost, so you know what to fix.
"""
from __future__ import annotations

import pandas as pd


def implementation_shortfall(paper_curve: pd.Series, real_curve: pd.Series) -> dict[str, float]:
    """Compare cumulative paper vs real equity curves.

    Both are cumulative-return series (start at 1.0). Returns total shortfall and
    an annualized figure.
    """
    paper_total = float(paper_curve.iloc[-1] / paper_curve.iloc[0] - 1)
    real_total = float(real_curve.iloc[-1] / real_curve.iloc[0] - 1)
    shortfall = paper_total - real_total
    n = len(real_curve)
    years = n / 252 if n else 1.0
    return {
        "paper_total_return": paper_total,
        "real_total_return": real_total,
        "implementation_shortfall": shortfall,
        "shortfall_annualized": shortfall / years if years > 0 else shortfall,
    }
