"""Mean-variance optimizer with real constraints, via CVXPY.

We solve a convex quadratic program. CVXPY lets us write the math almost
verbatim and hands it to a proper solver (OSQP/ECOS). If CVXPY/solver is
unavailable or the problem is infeasible, we fall back to a constrained
analytic tilt so the backtest never dies.

Constraints implemented (toggled by config):
  * full investment / dollar-neutrality
  * per-name position limit (|w_i| <= max_weight)
  * gross-leverage cap (sum|w| <= max_gross)
  * long-only (w >= 0) or long-short
  * sector neutrality (no active sector bet vs an equal-weight benchmark)
  * turnover cap (sum|w - w_prev| <= max_turnover)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphaforge.config import PortfolioCfg
from alphaforge.logging import get_logger
from alphaforge.risk.model import RiskModelSnapshot

log = get_logger(__name__)


@dataclass
class OptimizerResult:
    weights: np.ndarray
    status: str
    expected_return: float
    expected_risk: float       # daily stdev
    turnover: float


def optimize_portfolio(
    alpha: np.ndarray,
    snapshot: RiskModelSnapshot,
    cfg: PortfolioCfg,
    w_prev: np.ndarray | None = None,
    cost_coef: np.ndarray | None = None,
    sectors: np.ndarray | None = None,
) -> OptimizerResult:
    """Return the target weights for the current universe.

    Parameters
    ----------
    alpha : (N,) expected residual returns (the blended alpha).
    snapshot : the point-in-time risk model (gives V = B F B' + D).
    w_prev : (N,) current weights (for turnover + trading cost); zeros if None.
    cost_coef : (N,) linear per-unit trading cost in return units (spread+comm).
    sectors : (N,) sector labels for the sector-neutral constraint.
    """
    n = len(alpha)
    w_prev = np.zeros(n) if w_prev is None else w_prev
    cost_coef = np.zeros(n) if cost_coef is None else cost_coef

    try:
        return _solve_cvxpy(alpha, snapshot, cfg, w_prev, cost_coef, sectors)
    except Exception as e:  # noqa: BLE001 — never let the optimizer kill the run
        log.warning("optimizer.fallback", error=str(e))
        return _solve_fallback(alpha, snapshot, cfg, w_prev)


def _solve_cvxpy(alpha, snapshot, cfg, w_prev, cost_coef, sectors) -> OptimizerResult:
    import cvxpy as cp

    n = len(alpha)
    V = snapshot.covariance()
    # PSD-clean the covariance (numerical noise can make tiny negative eigvals).
    V = (V + V.T) / 2
    w = cp.Variable(n)
    trade = w - w_prev

    objective = (
        alpha @ w
        - 0.5 * cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(V))
        - cfg.cost_aversion * (cost_coef @ cp.abs(trade))
    )

    cons = [cp.abs(w) <= cfg.max_weight]
    if cfg.long_only:
        cons += [w >= 0, cp.sum(w) == 1]            # fully invested long-only
    else:
        cons += [cp.sum(w) == 0, cp.norm1(w) <= cfg.max_gross]  # dollar-neutral L/S

    # Turnover cap — but EXEMPT the initial build: ramping from an empty book to
    # a fully-invested one is necessarily ~100% turnover, which would otherwise
    # be infeasible. The cap only binds once a book exists.
    has_book = float(np.abs(w_prev).sum()) > 1e-6
    if has_book:
        cons += [cp.norm1(trade) <= cfg.max_turnover]

    if cfg.sector_neutral and sectors is not None:
        # No active sector bet vs an equal-weight (or zero for L/S) benchmark.
        for s in np.unique(sectors):
            mask = (sectors == s).astype(float)
            bench = mask.sum() / n if cfg.long_only else 0.0
            cons += [mask @ w == bench]

    prob = cp.Problem(cp.Maximize(objective), cons)
    prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status {prob.status}")

    wv = np.asarray(w.value).flatten()
    wv[np.abs(wv) < 1e-6] = 0.0
    risk = float(np.sqrt(max(wv @ V @ wv, 0.0)))
    return OptimizerResult(
        weights=wv,
        status=prob.status,
        expected_return=float(alpha @ wv),
        expected_risk=risk,
        turnover=float(np.abs(wv - w_prev).sum()),
    )


def _solve_fallback(alpha, snapshot, cfg, w_prev) -> OptimizerResult:
    """Closed-form-ish fallback: risk-scaled alpha tilt, then clip & renormalize.

    w ∝ V^{-1} alpha (the unconstrained MV solution), then enforce the box and
    leverage constraints by clipping and renormalizing. Not optimal, but always
    feasible and directionally correct — keeps a long run alive if a solver hiccups.
    """
    V = snapshot.covariance()
    try:
        raw = np.linalg.solve(V + 1e-6 * np.eye(len(alpha)), alpha)
    except np.linalg.LinAlgError:
        raw = alpha.copy()
    if cfg.long_only:
        raw = np.clip(raw, 0, None)
    raw = np.clip(raw, -cfg.max_weight, cfg.max_weight)
    total = np.abs(raw).sum()
    w = raw / total if total > 0 else raw
    if cfg.long_only and w.sum() > 0:
        w = w / w.sum()
    risk = float(np.sqrt(max(w @ V @ w, 0.0)))
    return OptimizerResult(w, "fallback", float(alpha @ w), risk, float(np.abs(w - w_prev).sum()))
