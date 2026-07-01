"""Signals must be point-in-time and well-formed (no look-ahead, no all-NaN)."""
from __future__ import annotations

import numpy as np

from alphaforge.signals.library import compute_all_signals


def test_signals_present_and_finite(lake):
    panel = compute_all_signals(lake)
    for col in ["value", "momentum", "size", "quality", "low_vol", "altdata"]:
        assert col in panel.columns
    # Each signal should have real (non-all-NaN) coverage once warmed up.
    warm = panel[panel["date"] > panel["date"].min() + np.timedelta64(300, "D")]
    for col in ["momentum", "low_vol", "size"]:
        assert warm[col].notna().mean() > 0.8, f"{col} too sparse"


def test_momentum_is_lagged(lake):
    """12-1 momentum needs ~252 days of history, so early dates must be NaN."""
    panel = compute_all_signals(lake).sort_values(["security_id", "date"])
    first_dates = panel.groupby("security_id").head(200)
    # In the first 200 days no name can have a full 12-1 momentum value.
    assert first_dates["momentum"].notna().mean() < 0.05
