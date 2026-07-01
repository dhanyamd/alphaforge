"""Factor decay monitoring — why the research loop never ends.

A factor's edge erodes as others discover it. We track each factor's information
coefficient (IC) through time (rolling cross-sectional rank correlation between
the signal and the forward return). A factor whose rolling IC is trending toward
zero is *decaying* — the signal the performance block sends back to research,
saying "this one needs a refresh, or retirement."
"""
from __future__ import annotations

import pandas as pd


def rolling_ic(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    signal_cols: list[str],
    window: int = 63,
) -> pd.DataFrame:
    """Daily cross-sectional rank IC per factor, smoothed over a rolling window.

    Returns a frame indexed by date with one column per factor = its smoothed IC.
    """
    df = signals.merge(forward_returns, on=["date", "security_id"], how="inner")
    daily = []
    for date, cross in df.groupby("date"):
        row = {"date": date}
        fr = cross["fwd_ret"].rank()
        if fr.std(ddof=0) == 0 or len(fr) < 5:
            continue
        for col in signal_cols:
            if col not in cross:
                continue
            sr = cross[col].rank()
            # Guard a constant cross-section (std 0 -> corr is NaN/divide warning).
            row[col] = float(sr.corr(fr)) if sr.std(ddof=0) > 0 else 0.0
        daily.append(row)
    ic = pd.DataFrame(daily).set_index("date").sort_index()
    return ic.rolling(window, min_periods=window // 3).mean()


def decay_report(ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize each factor's IC: full-sample mean vs the most recent window.

    A large drop from `mean_ic` to `recent_ic` flags a decaying factor — exactly
    the low-hanging fruit a portfolio manager green-lights for a second research
    project.
    """
    if ic.empty:
        return pd.DataFrame()
    mean_ic = ic.mean()
    recent_ic = ic.tail(63).mean()
    out = pd.DataFrame({"mean_ic": mean_ic, "recent_ic": recent_ic})
    out["decay"] = out["recent_ic"] - out["mean_ic"]
    out["status"] = out["decay"].apply(lambda d: "DECAYING" if d < -0.01 else "stable")
    return out.sort_values("recent_ic", ascending=False)
