"""Risk model produces a valid covariance; cost model is monotone in size."""
from __future__ import annotations

import numpy as np

from alphaforge.config import ExecutionCfg
from alphaforge.execution.costs import CostModel
from alphaforge.risk.model import RiskModelSnapshot, annualize_risk


def test_covariance_is_psd():
    rng = np.random.default_rng(0)
    n, k = 25, 5
    B = rng.normal(0, 1, (n, k))
    F = np.diag(rng.uniform(1e-4, 4e-4, k))
    d = rng.uniform(1e-4, 9e-4, n)
    snap = RiskModelSnapshot(None, [f"S{i}" for i in range(n)], [f"f{j}" for j in range(k)], B, F, d)
    V = snap.covariance()
    eig = np.linalg.eigvalsh((V + V.T) / 2)
    assert eig.min() > -1e-10, "covariance not PSD"


def test_sqrt_time_scaling():
    # Annualizing daily vol multiplies by sqrt(252), not 252.
    assert abs(annualize_risk(0.01) - 0.01 * np.sqrt(252)) < 1e-12


def test_market_impact_monotone_in_size():
    cm = CostModel(ExecutionCfg())
    adv = np.array([1e7])
    vol = np.array([0.02])
    small = cm.realized_cost(np.array([1e5]), adv, vol)[0]
    large = cm.realized_cost(np.array([1e6]), adv, vol)[0]
    # Bigger order -> strictly higher total cost (square-root impact term grows).
    assert large > small
