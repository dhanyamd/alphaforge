"""ASOF join must be point-in-time: it may only see the PAST, never the future."""
from __future__ import annotations

import pandas as pd

from alphaforge.storage.asof import asof_join


def test_asof_takes_last_known_past_value():
    # A quote table that updates on day 1 and day 5.
    quotes = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-05"]),
        "security_id": ["A", "A"],
        "price": [10.0, 20.0],
    })
    # A trade on day 3 must see the day-1 quote (10), NOT the future day-5 quote.
    trades = pd.DataFrame({"date": pd.to_datetime(["2020-01-03"]), "security_id": ["A"]})
    out = asof_join(trades, quotes, on="date", by="security_id", direction="backward")
    assert out["price"].iloc[0] == 10.0


def test_asof_no_future_leak_before_first_obs():
    quotes = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-10"]),
        "security_id": ["A"],
        "price": [99.0],
    })
    trades = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "security_id": ["A"]})
    out = asof_join(trades, quotes, on="date", by="security_id", direction="backward")
    # No past observation exists -> must be null, never the future 99.
    assert pd.isna(out["price"].iloc[0])
