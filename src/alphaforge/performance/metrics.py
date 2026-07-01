"""Performance metrics — IR, Sharpe, drawdown, and the fundamental law.

All built on a daily return series. The headline is the **information ratio**:
annualized mean active return divided by annualized active risk. We also report
the fundamental-law decomposition so you can see *why* the IR is what it is:
IR ≈ IC · sqrt(breadth), where breadth is the number of independent bets/year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / TRADING_DAYS
    if r.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def information_ratio(active_returns: pd.Series) -> float:
    """IR = annualized mean active return / annualized active risk.

    For a benchmark-relative strategy `active_returns` is (portfolio - benchmark).
    For an absolute strategy it's just the return series.
    """
    r = active_returns.dropna()
    if r.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=0) * np.sqrt(TRADING_DAYS))


def fundamental_law(mean_ic: float, breadth: int) -> float:
    """IR ≈ IC · sqrt(breadth). breadth = # independent bets per year."""
    return float(mean_ic * np.sqrt(max(breadth, 0)))


def performance_summary(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    mean_ic: float | None = None,
    breadth: int | None = None,
) -> dict[str, float]:
    """One dict with every headline metric for the tearsheet."""
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    active = returns - benchmark.reindex(returns.index).fillna(0.0) if benchmark is not None else returns

    out = {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe(returns),
        "information_ratio": information_ratio(active),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((returns > 0).mean()),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
    }
    if mean_ic is not None and breadth is not None:
        out["mean_ic"] = mean_ic
        out["breadth"] = float(breadth)
        out["ir_implied_by_law"] = fundamental_law(mean_ic, breadth)
    return out
