"""The signal library — value, momentum, size, quality, low-vol, alt-data.

Each function returns a long frame (date, security_id, <signal>) built ONLY from
information knowable as of each date:

  * **momentum** : trailing 12-month return skipping the most recent month
    (the classic 12-1 momentum; skipping the last month avoids short-term
    reversal contamination).
  * **size**     : -log(market cap). Small-cap tilt → negative sign.
  * **low_vol**  : -trailing 63-day return volatility. Low-vol tilt → negative sign.
  * **value**    : book-to-price, as-of-joined from fundamentals with the
    reporting lag respected (no look-ahead).
  * **quality**  : gross profitability (gross profit / total assets).
  * **altdata**  : the alt-data score, forward-filled from its weekly publish.

Price-based signals are computed with pandas rolling windows; fundamentals/alt
are attached point-in-time via an ASOF join.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.storage.asof import asof_join
from alphaforge.storage.warehouse import Warehouse

SIGNALS = ["value", "momentum", "size", "quality", "low_vol", "altdata"]


def _prices(wh: Warehouse) -> pd.DataFrame:
    px = wh.read_table("prices", columns=["date", "security_id", "close", "ret", "mktcap"])
    px["date"] = pd.to_datetime(px["date"])
    return px.sort_values(["security_id", "date"])


def momentum(px: pd.DataFrame) -> pd.DataFrame:
    """12-1 month momentum: cumulative return from t-252 to t-21."""
    g = px.groupby("security_id")["close"]
    # close_{t-21} / close_{t-252} - 1, all knowable at t.
    px = px.copy()
    px["c_21"] = g.shift(21)
    px["c_252"] = g.shift(252)
    px["momentum"] = px["c_21"] / px["c_252"] - 1.0
    return px[["date", "security_id", "momentum"]]


def size(px: pd.DataFrame) -> pd.DataFrame:
    px = px.copy()
    px["size"] = -np.log(px["mktcap"].clip(lower=1.0))
    return px[["date", "security_id", "size"]]


def low_vol(px: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    px = px.copy()
    px["vol"] = px.groupby("security_id")["ret"].transform(
        lambda s: s.rolling(window, min_periods=20).std()
    )
    px["low_vol"] = -px["vol"]
    return px[["date", "security_id", "low_vol"]]


# fundamentals / alt
def _asof_attach(px_dates: pd.DataFrame, vendor: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Attach a vendor value to every (date, security_id) via point-in-time ASOF.

    For each trading date we take the most recent vendor observation at or before
    that date — i.e. only what was *published and knowable* by then. This is the
    look-ahead firewall for slow-moving fundamentals and weekly alt-data.
    """
    left = px_dates[["date", "security_id"]].sort_values("date")
    right = vendor[["date", "security_id", value_col]].dropna(subset=[value_col]).sort_values("date")
    out = asof_join(left, right, on="date", by="security_id", direction="backward")
    return out[["date", "security_id", value_col]]


def value(wh: Warehouse, px_dates: pd.DataFrame) -> pd.DataFrame:
    funda = wh.read_table("fundamentals", columns=["date", "security_id", "book_to_price"])
    funda["date"] = pd.to_datetime(funda["date"])
    out = _asof_attach(px_dates, funda, "book_to_price").rename(columns={"book_to_price": "value"})
    return out


def quality(wh: Warehouse, px_dates: pd.DataFrame) -> pd.DataFrame:
    funda = wh.read_table("fundamentals", columns=["date", "security_id", "gross_profitability"])
    funda["date"] = pd.to_datetime(funda["date"])
    out = _asof_attach(px_dates, funda, "gross_profitability").rename(
        columns={"gross_profitability": "quality"}
    )
    return out


def altdata(wh: Warehouse, px_dates: pd.DataFrame) -> pd.DataFrame:
    alt = wh.read_table("altdata", columns=["date", "security_id", "alt_score"])
    alt["date"] = pd.to_datetime(alt["date"])
    out = _asof_attach(px_dates, alt, "alt_score").rename(columns={"alt_score": "altdata"})
    return out


# combine
def compute_all_signals(wh: Warehouse) -> pd.DataFrame:
    """Compute every raw signal and return one wide panel: date, security_id, <signals>.

    These are the RAW signals (un-cleaned). The alpha layer applies cross-
    sectional hygiene (winsorize/neutralize/z-score) and the skill-scaling.
    """
    px = _prices(wh)
    px_dates = px[["date", "security_id"]].drop_duplicates()

    frames = [
        momentum(px),
        size(px),
        low_vol(px),
        value(wh, px_dates),
        quality(wh, px_dates),
        altdata(wh, px_dates),
    ]
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on=["date", "security_id"], how="left")
    return panel.sort_values(["date", "security_id"]).reset_index(drop=True)
