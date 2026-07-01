"""Transaction-cost model — commission + spread + square-root market impact.

The square-root law (Almgren et al., empirically robust for liquid equities):

    impact_fraction ≈ η · σ_daily · sqrt(Q / ADV)

where Q is shares traded, ADV is average daily volume, σ_daily is the name's
daily volatility, and η is a calibration coefficient. Intuition: trading a small
slice of a name's daily volume is nearly free; trading a large fraction moves the
price against you, and the cost grows with the *square root* of participation —
not linearly — which is why you slice big orders over time.

Two entry points:
  * `linear_cost_coef`  — the convex (commission+spread) cost per unit of weight
    traded, fed to the optimizer (keeps the QP convex).
  * `realized_cost`     — the FULL cost incl. nonlinear impact, applied by the
    simulator to actual fills.
"""
from __future__ import annotations

import numpy as np

from alphaforge.config import ExecutionCfg

BPS = 1e-4


class CostModel:
    def __init__(self, cfg: ExecutionCfg) -> None:
        self.cfg = cfg

    def linear_cost_coef(self) -> float:
        """Per-dollar linear cost (commission + half-spread), in return fraction.

        This is the part of cost that's linear in trade size, so it stays convex
        in the optimizer. Returned as a fraction (e.g. 0.0003 = 3 bps round-ish).
        """
        return (self.cfg.commission_bps + self.cfg.half_spread_bps) * BPS

    def impact_fraction(self, trade_dollars: np.ndarray, adv_dollars: np.ndarray,
                        daily_vol: np.ndarray) -> np.ndarray:
        """Square-root market-impact cost as a fraction of traded notional."""
        adv = np.where(adv_dollars > 0, adv_dollars, np.nan)
        participation = np.abs(trade_dollars) / adv
        participation = np.nan_to_num(participation, nan=0.0)
        impact = self.cfg.impact_coef * daily_vol * np.sqrt(participation)
        return np.nan_to_num(impact, nan=0.0)

    def realized_cost(self, trade_dollars: np.ndarray, adv_dollars: np.ndarray,
                      daily_vol: np.ndarray) -> np.ndarray:
        """Total $ cost per name = (commission + spread + impact) · |trade$|."""
        linear = self.linear_cost_coef()
        impact = self.impact_fraction(trade_dollars, adv_dollars, daily_vol)
        return (linear + impact) * np.abs(trade_dollars)
