"""End-to-end edge test: the model's alpha must actually predict forward returns.

Unlike the plumbing tests, this one asserts the strategy has REAL predictive
power on the (seeded, reproducible) synthetic market — a positive IC and a
top-beats-bottom quantile spread. If a future change silently breaks the signal
chain or injects look-ahead in the wrong direction, this test catches it.
"""
from __future__ import annotations

from alphaforge.alpha.combine import build_alphas
from alphaforge.performance.predictive import analyze_predictive_power


def test_alpha_has_positive_ic_and_quantile_spread(lake, small_cfg):
    small_cfg.storage.lake_path = lake.root  # ensure same lake
    alpha_panel = build_alphas(small_cfg, lake)
    prices = lake.read_table("prices", columns=["date", "security_id", "ret"])
    rep = analyze_predictive_power(alpha_panel, prices)

    # The blended alpha must predict: positive IC, positive on most days.
    assert rep.mean_ic > 0.0, f"alpha has no predictive power (IC={rep.mean_ic:.4f})"
    assert rep.pct_days_positive > 0.5, "IC positive on fewer than half the days"

    # The names it liked (top quintile) must beat the names it disliked (bottom).
    assert rep.top_minus_bottom_bps > 0.0, "top quintile did not beat bottom quintile"
