"""The optimizer must respect its constraints — limits, leverage, long-only."""
from __future__ import annotations

import numpy as np

from alphaforge.config import PortfolioCfg
from alphaforge.portfolio.optimizer import optimize_portfolio
from alphaforge.risk.model import RiskModelSnapshot


def _toy_snapshot(n=30, k=4, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (n, k))
    F = np.diag(rng.uniform(1e-4, 4e-4, k))
    d = rng.uniform(1e-4, 9e-4, n)
    return RiskModelSnapshot(
        date=None, security_ids=[f"S{i}" for i in range(n)],
        factor_names=[f"f{j}" for j in range(k)], B=B, F=F, d=d,
    )


def test_long_only_and_position_limits():
    snap = _toy_snapshot()
    n = len(snap.security_ids)
    alpha = np.random.default_rng(1).normal(0, 0.001, n)
    cfg = PortfolioCfg(max_weight=0.05, long_only=True, sector_neutral=False, max_turnover=1.0)
    res = optimize_portfolio(alpha, snap, cfg, w_prev=np.zeros(n))
    w = res.weights
    assert (w >= -1e-6).all(), "long-only violated"
    assert w.max() <= cfg.max_weight + 1e-4, "position limit violated"
    assert abs(w.sum() - 1.0) < 1e-3, "not fully invested"


def test_market_neutral_sums_to_zero():
    snap = _toy_snapshot()
    n = len(snap.security_ids)
    alpha = np.random.default_rng(2).normal(0, 0.001, n)
    cfg = PortfolioCfg(max_weight=0.10, long_only=False, sector_neutral=False,
                       max_gross=1.0, max_turnover=2.0)
    res = optimize_portfolio(alpha, snap, cfg, w_prev=np.zeros(n))
    assert abs(res.weights.sum()) < 1e-2, "dollar-neutral violated"
    assert np.abs(res.weights).sum() <= cfg.max_gross + 1e-2, "gross leverage violated"
