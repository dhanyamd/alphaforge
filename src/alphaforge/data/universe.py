"""Universe definition — "define your universe first".

Your instinct is to trade everything for maximum breadth, but illiquid names you
can't get in and out of without moving the price against yourself destroy more
value than they add. So before any signal runs you draw a sensible boundary: a
point-in-time set of tradable names, gated on liquidity (average dollar volume).

The universe is itself point-in-time — a name enters when it becomes liquid
enough and leaves when it doesn't, so the backtest never trades something it
couldn't actually have traded that day.
"""
from __future__ import annotations

import pandas as pd

from alphaforge.config import Config
from alphaforge.logging import get_logger
from alphaforge.storage.warehouse import Warehouse

log = get_logger(__name__)


def build_universe(cfg: Config, wh: Warehouse, adv_window: int = 21) -> pd.DataFrame:
    """Return a long frame (date, security_id, in_universe) gated on liquidity.

    A name is in-universe on date D if its trailing `adv_window`-day average
    dollar volume exceeds `min_adv_usd`; we additionally keep only the top
    `top_by_adv` names each day to cap the book at the most liquid set.
    """
    px = wh.read_table("prices", columns=["date", "security_id", "dollar_volume"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["security_id", "date"])

    # Trailing ADV (average daily dollar volume) per name.
    px["adv"] = (
        px.groupby("security_id")["dollar_volume"]
        .transform(lambda s: s.rolling(adv_window, min_periods=5).mean())
    )

    liquid = px["adv"] >= cfg.universe.min_adv_usd
    px["in_universe"] = liquid

    # Keep the top-N most liquid names each day.
    px["adv_rank"] = px.groupby("date")["adv"].rank(ascending=False, method="first")
    px.loc[px["adv_rank"] > cfg.universe.top_by_adv, "in_universe"] = False

    uni = px[["date", "security_id", "adv", "in_universe"]].copy()
    daily_count = uni[uni["in_universe"]].groupby("date")["security_id"].nunique()
    log.info(
        "universe.built",
        median_size=int(daily_count.median()) if len(daily_count) else 0,
        min_adv_usd=cfg.universe.min_adv_usd,
    )
    return uni
