"""The backtest engine — wires every block into one daily loop.

This is the "one connected machine": data -> signals -> alpha -> risk ->
portfolio construction -> implementation -> performance, run forward through
history one rebalance at a time, with strict point-in-time discipline so no
future information leaks backward.
"""

from alphaforge.backtest.engine import BacktestResult, run_backtest

__all__ = ["run_backtest", "BacktestResult"]
