"""(V) IMPLEMENTATION + TRADING — subtract as little value as possible.

The optimizer hands you a target; getting there leaks alpha through four costs:
  * commission  — per-share fee to the broker,
  * spread      — you buy at the ask, sell at the bid,
  * market impact — buying size pushes the price against you (the big sneaky one;
    "you can't observe the market without disturbing it"),
  * opportunity cost — the fill you waited on that ran away.

We model commission + spread + a square-root market-impact term, simulate fills,
and measure the total via **implementation shortfall**: the gap between a
frictionless paper portfolio and the real one.
"""

from alphaforge.execution.costs import CostModel
from alphaforge.execution.shortfall import implementation_shortfall

__all__ = ["CostModel", "implementation_shortfall"]
