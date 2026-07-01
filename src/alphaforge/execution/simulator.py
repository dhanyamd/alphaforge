"""Fill simulator — turn target weights into realized trades and costs.

We model a single-shot rebalance with a participation cap: you cannot trade more
than `participation` × ADV of a name in one go (trying to dump 50% of daily
volume is unrealistic and ruinously expensive). Any unfilled remainder carries to
the next rebalance — that residual is a source of opportunity cost.

Returns the realized post-trade weights and the total cost in return units, which
the backtest subtracts from gross P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphaforge.execution.costs import CostModel


@dataclass
class FillResult:
    realized_weights: np.ndarray   # weights actually achieved after participation cap
    cost_return: float             # total trading cost as a fraction of portfolio value
    turnover: float
    capped_names: int              # how many orders hit the participation limit


def simulate_fills(
    w_target: np.ndarray,
    w_prev: np.ndarray,
    adv_dollars: np.ndarray,
    daily_vol: np.ndarray,
    portfolio_value: float,
    cost_model: CostModel,
    participation: float,
) -> FillResult:
    """Simulate filling from `w_prev` toward `w_target` under a participation cap."""
    desired_trade_w = w_target - w_prev
    desired_trade_dollars = desired_trade_w * portfolio_value

    # Participation cap: max tradable notional per name this rebalance.
    max_trade_dollars = participation * adv_dollars
    filled_dollars = np.sign(desired_trade_dollars) * np.minimum(
        np.abs(desired_trade_dollars), max_trade_dollars
    )
    capped = int(np.sum(np.abs(desired_trade_dollars) > max_trade_dollars + 1e-9))

    realized_weights = w_prev + filled_dollars / portfolio_value
    cost_dollars = cost_model.realized_cost(filled_dollars, adv_dollars, daily_vol).sum()
    cost_return = cost_dollars / portfolio_value if portfolio_value > 0 else 0.0
    turnover = float(np.abs(filled_dollars).sum() / portfolio_value) if portfolio_value > 0 else 0.0

    return FillResult(realized_weights, cost_return, turnover, capped)
