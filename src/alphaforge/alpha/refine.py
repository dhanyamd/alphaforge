"""Signal -> alpha refinement (Grinold) + empirical IC estimation.

`alpha_i = vol_i * IC * score_i`  (per name, per date)

`score` is the cleaned (winsorized/neutralized/z-scored) signal. `vol` is the
name's forecast volatility (we use trailing realized vol as a cheap proxy). `IC`
is the signal's skill — we both accept a prior from config AND measure it
empirically out-of-sample so you can see a real decaying IC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.signals.base import clean_signal


def clean_panel(
    signals: pd.DataFrame,
    static: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """Apply cross-sectional hygiene to every signal, per date.

    Returns the panel with each `<sig>` replaced by its cleaned z-score, sector-
    and size-neutralized so the factors are independent bets.
    """
    sec_map = static.set_index("security_id")["sector"]
    out_rows = []
    for date, cross in signals.groupby("date"):
        cross = cross.copy()
        sector = cross["security_id"].map(sec_map)
        size_raw = cross["size"] if "size" in cross else None
        for col in signal_cols:
            if col not in cross:
                continue
            raw = cross[col]
            # Skip neutralizing 'size' on itself.
            sz = None if col == "size" else size_raw
            cross[col] = clean_signal(raw.fillna(raw.median()), sector=sector, size=sz).values
        out_rows.append(cross)
    return pd.concat(out_rows, ignore_index=True)


def forecast_vol(wh_prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Per-name forecast volatility = trailing realized daily vol (annualized off)."""
    px = wh_prices.sort_values(["security_id", "date"]).copy()
    px["fvol"] = px.groupby("security_id")["ret"].transform(
        lambda s: s.rolling(window, min_periods=20).std()
    )
    return px[["date", "security_id", "fvol"]]


def empirical_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """Rank IC = Spearman corr between a signal and the next-period return.

    IC is THE measure of signal skill. Typical real single-name equity ICs are
    tiny — 0.02 to 0.05 — and that's fine; the fundamental law says breadth turns
    small skill into a real information ratio.
    """
    s = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(s) < 10:
        return np.nan
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))


def refine(score: pd.Series, vol: pd.Series, ic: float) -> pd.Series:
    """alpha = vol * IC * score, the Grinold refinement (residual-return units)."""
    return vol.fillna(vol.median()) * ic * score
