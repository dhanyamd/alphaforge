"""The end-to-end backtest — run the whole multifactor machine through history.

For each rebalance date (point-in-time, no look-ahead):
  1. take the tradable universe as of that date,
  2. read the blended alpha for those names,
  3. build the point-in-time risk model snapshot (B, F, d),
  4. optimize the target book (return - risk penalty - cost, with constraints),
  5. simulate fills + costs (participation-capped),
  6. hold to the next rebalance, accruing daily P&L (weights drift with returns).

We run TWO books in parallel — a frictionless "paper" book and the real
cost-bearing book — so we can read off the implementation shortfall. At the end
we compute performance metrics, factor attribution, and factor-IC decay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.alpha.combine import build_alpha_panel
from alphaforge.config import Config
from alphaforge.data.universe import build_universe
from alphaforge.execution.costs import CostModel
from alphaforge.execution.shortfall import implementation_shortfall
from alphaforge.execution.simulator import simulate_fills
from alphaforge.logging import get_logger
from alphaforge.performance.decay import decay_report, rolling_ic
from alphaforge.performance.metrics import performance_summary
from alphaforge.portfolio.optimizer import optimize_portfolio
from alphaforge.risk.model import RiskModel
from alphaforge.storage.warehouse import Warehouse

log = get_logger(__name__)

# Notional capital. Sets the scale of trades vs ADV, which drives participation
# and market impact. Real number; pick something a small fund would run.
NOTIONAL_AUM = 50_000_000.0


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    paper_returns: pd.Series
    benchmark_returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]
    shortfall: dict[str, float]
    decay: pd.DataFrame
    weights_history: pd.DataFrame
    diagnostics: dict[str, float] = field(default_factory=dict)


def run_backtest(cfg: Config, wh: Warehouse) -> BacktestResult:
    rng_log = log.bind(run=cfg.run.name)
    rng_log.info("backtest.start", start=cfg.run.start_date, end=cfg.run.end_date)

    # ---- 1. assemble all inputs (these are the upstream pipeline stages) ----
    # Picks the linear or gradient-boosted alpha model from config (alpha.model).
    alpha_panel = build_alpha_panel(cfg, wh)
    static = wh.read_table("security_static")
    prices = wh.read_table(
        "prices", columns=["date", "security_id", "ret", "close", "mktcap", "dollar_volume"]
    )
    prices["date"] = pd.to_datetime(prices["date"])
    universe = build_universe(cfg, wh)

    # Fit the factor risk model once (each day's regression uses only that day).
    risk_model = RiskModel(cfg).fit(alpha_panel, prices, static)

    # ---- 2. wide matrices for a fast daily P&L loop -------------------------
    ret_wide = prices.pivot(index="date", columns="security_id", values="ret").sort_index()
    all_dates = ret_wide.index
    sec_map = static.set_index("security_id")["sector"]

    # Rebalance calendar (e.g. weekly Fridays) intersected with trading days.
    rebal_dates = _rebalance_calendar(all_dates, cfg.run.rebalance)

    cost_model = CostModel(cfg.execution)

    # State for the real and paper books (weight vectors keyed by security_id).
    real_w = pd.Series(0.0, index=ret_wide.columns)
    paper_w = pd.Series(0.0, index=ret_wide.columns)

    daily_real: list[tuple[pd.Timestamp, float]] = []
    daily_paper: list[tuple[pd.Timestamp, float]] = []
    daily_bench: list[tuple[pd.Timestamp, float]] = []
    weights_hist: list[pd.DataFrame] = []
    capped_total = 0

    # Pre-index alpha and universe for quick per-date lookup.
    alpha_by_date = {d: g for d, g in alpha_panel.groupby("date")}
    uni_by_date = {d: set(g.loc[g["in_universe"], "security_id"]) for d, g in universe.groupby("date")}

    next_rebal = set(rebal_dates)
    last_rebal_date: pd.Timestamp | None = None
    # Trading cost incurred at a rebalance is charged to the NEXT day's return
    # (you trade at today's close; the drag shows up tomorrow). Charging the
    # return stream — not the weights — is correct, because weights get
    # renormalized by drift each day and would otherwise wash the cost out.
    pending_cost = 0.0

    for i, date in enumerate(all_dates):
        # --- accrue P&L from holding yesterday's weights into today ----------
        if i > 0:
            r = ret_wide.loc[date].fillna(0.0)
            real_ret = float((real_w * r).sum()) - pending_cost
            paper_ret = float((paper_w * r).sum())   # frictionless book: no cost
            pending_cost = 0.0
            daily_real.append((date, real_ret))
            daily_paper.append((date, paper_ret))
            # Benchmark = equal-weight current universe (the "buy the index" baseline).
            uni_ids = uni_by_date.get(last_rebal_date, set()) if last_rebal_date else set()
            bench_ret = float(r[list(uni_ids)].mean()) if uni_ids else 0.0
            daily_bench.append((date, bench_ret))
            # Drift weights with realized returns (positions grow/shrink with price).
            real_w = _drift(real_w, r)
            paper_w = _drift(paper_w, r)

        # --- rebalance? ------------------------------------------------------
        if date in next_rebal:
            res = _rebalance(
                date, cfg, alpha_by_date, uni_by_date, risk_model, sec_map,
                prices, universe, cost_model, real_w, paper_w, ret_wide.columns,
            )
            if res is not None:
                real_w, paper_w, capped, cost_return = res
                capped_total += capped
                pending_cost += cost_return        # charged to tomorrow's return
                last_rebal_date = date
                weights_hist.append(
                    pd.DataFrame({"date": date, "security_id": real_w.index, "weight": real_w.values})
                )

    # ---- 3. assemble curves + metrics --------------------------------------
    real_returns = _to_series(daily_real)
    paper_returns = _to_series(daily_paper)
    bench_returns = _to_series(daily_bench)
    equity = (1 + real_returns).cumprod()
    paper_equity = (1 + paper_returns).cumprod()

    # Factor-IC decay (the research-loop feedback signal).
    fwd = prices.copy()
    fwd["fwd_ret"] = fwd.groupby("security_id")["ret"].shift(-1)
    signal_cols = [s.name for s in cfg.enabled_signals]
    ic = rolling_ic(
        alpha_panel[["date", "security_id", *signal_cols]],
        fwd[["date", "security_id", "fwd_ret"]],
        signal_cols,
    )
    decay = decay_report(ic)

    # Breadth ≈ names traded × rebalances/year (independent-ish bets per year).
    yrs = max(len(real_returns) / 252, 1e-9)
    breadth = int(cfg.universe.top_by_adv * (len(rebal_dates) / yrs))
    # Fundamental-law IC must be the IC of the COMBINED alpha (what we actually
    # trade), not the average of per-factor ICs (which cancel out). Measure the
    # rank-corr of the blended alpha vs forward return.
    alpha_ic = rolling_ic(
        alpha_panel[["date", "security_id", "alpha"]],
        fwd[["date", "security_id", "fwd_ret"]],
        ["alpha"],
    )
    mean_ic = float(alpha_ic["alpha"].mean()) if not alpha_ic.empty else 0.0

    metrics = performance_summary(real_returns, benchmark=bench_returns, mean_ic=mean_ic, breadth=breadth)
    shortfall = implementation_shortfall(paper_equity, equity)

    result = BacktestResult(
        daily_returns=real_returns,
        paper_returns=paper_returns,
        benchmark_returns=bench_returns,
        equity_curve=equity,
        metrics=metrics,
        shortfall=shortfall,
        decay=decay,
        weights_history=pd.concat(weights_hist, ignore_index=True) if weights_hist else pd.DataFrame(),
        diagnostics={"participation_capped_orders": float(capped_total),
                     "rebalances": float(len(rebal_dates))},
    )
    rng_log.info("backtest.done", ir=round(metrics["information_ratio"], 3),
                 sharpe=round(metrics["sharpe"], 3), ann_ret=round(metrics["ann_return"], 4))
    return result


# --------------------------------------------------------------------------- helpers
def _rebalance(date, cfg, alpha_by_date, uni_by_date, risk_model, sec_map,
               prices, universe, cost_model, real_w, paper_w, all_cols):
    """Run one rebalance; return (new_real_w, new_paper_w, capped, cost_return) or None."""
    uni_ids = sorted(uni_by_date.get(date, set()))
    if len(uni_ids) < 20:
        return None
    apanel = alpha_by_date.get(date)
    if apanel is None:
        return None

    snap = risk_model.snapshot(date, uni_ids)
    ids = snap.security_ids                      # names with a full risk model row
    if len(ids) < 20:
        return None

    a = apanel.set_index("security_id")
    alpha = a.reindex(ids)["alpha"].fillna(0.0).values
    fvol = a.reindex(ids)["fvol"].fillna(a["fvol"].median()).values
    sectors = np.array([sec_map.get(s, "NA") for s in ids])

    # Liquidity (ADV in $) for participation/impact and the linear cost coef.
    uni_today = universe[(universe["date"] == date)].set_index("security_id")
    adv = uni_today.reindex(ids)["adv"].fillna(uni_today["adv"].median()).values
    cost_coef = np.full(len(ids), cost_model.linear_cost_coef())

    w_prev = real_w.reindex(ids).fillna(0.0).values

    # ---- optimize the target book ----
    opt = optimize_portfolio(alpha, snap, cfg.portfolio, w_prev=w_prev,
                             cost_coef=cost_coef, sectors=sectors)
    w_target = opt.weights

    # ---- simulate fills + costs on the real book ----
    fill = simulate_fills(
        w_target=w_target, w_prev=w_prev, adv_dollars=adv, daily_vol=fvol,
        portfolio_value=NOTIONAL_AUM, cost_model=cost_model,
        participation=cfg.execution.participation,
    )
    # Real book holds the (participation-capped) realized weights. The trading
    # cost is returned separately and charged to the next day's return stream —
    # NOT baked into weights, which get renormalized by drift and would wash it out.
    new_real = pd.Series(0.0, index=all_cols)
    new_real.loc[ids] = fill.realized_weights

    # Paper book: instant, frictionless fill to target (the no-cost ideal).
    new_paper = pd.Series(0.0, index=all_cols)
    new_paper.loc[ids] = w_target
    return new_real, new_paper, fill.capped_names, fill.cost_return


def _rebalance_calendar(dates: pd.DatetimeIndex, rule: str) -> list[pd.Timestamp]:
    """Map a pandas offset rule (e.g. 'W-FRI', 'B', 'ME') to actual trading days."""
    s = pd.Series(1, index=dates)
    if rule.upper() in ("B", "D"):
        return list(dates)
    grouped = s.resample(rule).last()
    # For each period take the last actual trading day on/before the period end.
    out = []
    for period_end in grouped.index:
        candidates = dates[dates <= period_end]
        if len(candidates):
            out.append(candidates[-1])
    return sorted(set(out))


def _drift(w: pd.Series, r: pd.Series) -> pd.Series:
    """Let weights drift with realized returns and renormalize to keep them weights."""
    grown = w * (1 + r.reindex(w.index).fillna(0.0))
    total = grown.abs().sum()
    return grown / total if total > 0 else grown


def _to_series(pairs: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    if not pairs:
        return pd.Series(dtype=float)
    idx, vals = zip(*pairs, strict=False)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))
