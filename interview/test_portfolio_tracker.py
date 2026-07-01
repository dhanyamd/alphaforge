"""Tests for the interview PMS — the cases an interviewer will probe."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_tracker import OrderStatus, OrderType, Portfolio, Side


def test_market_buy_updates_cash_and_position():
    pf = Portfolio(cash=100_000)
    pf.on_price("AAPL", 150.0)
    pf.submit_order("AAPL", Side.BUY, 100, OrderType.MARKET)
    assert pf.cash == 100_000 - 100 * 150
    assert pf.positions["AAPL"].quantity == 100
    assert pf.positions["AAPL"].avg_price == 150


def test_realized_pnl_on_sell():
    pf = Portfolio(cash=100_000)
    pf.on_price("AAPL", 150.0)
    pf.submit_order("AAPL", Side.BUY, 100, OrderType.MARKET)
    pf.on_price("AAPL", 170.0)
    pf.submit_order("AAPL", Side.SELL, 100, OrderType.MARKET)
    assert pf.positions["AAPL"].quantity == 0
    assert pf.realized_pnl() == (170 - 150) * 100      # +2000


def test_stop_loss_triggers_on_drop():
    pf = Portfolio(cash=100_000)
    pf.on_price("AAPL", 150.0)
    pf.submit_order("AAPL", Side.BUY, 100, OrderType.MARKET)
    stop = pf.submit_order("AAPL", Side.SELL, 100, OrderType.STOP, stop_price=140.0)
    assert stop.status is OrderStatus.OPEN
    pf.on_price("AAPL", 139.0)                          # crosses the stop
    assert stop.status is OrderStatus.FILLED
    assert pf.positions["AAPL"].quantity == 0
    assert pf.realized_pnl() == (139 - 150) * 100       # -1100 loss, as intended


def test_limit_buy_waits_for_price():
    pf = Portfolio(cash=100_000)
    pf.on_price("AAPL", 150.0)
    lim = pf.submit_order("AAPL", Side.BUY, 100, OrderType.LIMIT, limit_price=145.0)
    pf.on_price("AAPL", 148.0)                          # still above limit -> no fill
    assert lim.status is OrderStatus.OPEN
    pf.on_price("AAPL", 144.0)                          # drops through -> fills at 145
    assert lim.status is OrderStatus.FILLED
    assert lim.fill_price == 145.0


def test_average_cost_on_add():
    pf = Portfolio(cash=1_000_000)
    pf.on_price("X", 100.0); pf.submit_order("X", Side.BUY, 100, OrderType.MARKET)
    pf.on_price("X", 120.0); pf.submit_order("X", Side.BUY, 100, OrderType.MARKET)
    # 100@100 + 100@120 -> avg 110
    assert pf.positions["X"].avg_price == 110
    assert pf.positions["X"].quantity == 200


def test_equity_invariant():
    pf = Portfolio(cash=100_000)
    pf.on_price("AAPL", 150.0)
    pf.submit_order("AAPL", Side.BUY, 100, OrderType.MARKET)
    pf.on_price("AAPL", 165.0)
    # equity must equal initial cash + realized + unrealized, always.
    assert abs(pf.equity() - (pf.initial_cash + pf.realized_pnl() + pf.unrealized_pnl())) < 1e-9


def test_short_then_cover():
    pf = Portfolio(cash=100_000)
    pf.on_price("Z", 50.0)
    pf.submit_order("Z", Side.SELL, 100, OrderType.MARKET)   # open a SHORT
    assert pf.positions["Z"].quantity == -100
    pf.on_price("Z", 40.0)
    pf.submit_order("Z", Side.BUY, 100, OrderType.MARKET)    # cover lower -> profit
    assert pf.realized_pnl() == (50 - 40) * 100             # +1000
