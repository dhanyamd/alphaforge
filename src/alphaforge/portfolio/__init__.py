"""(IV) PORTFOLIO CONSTRUCTION — alpha + risk -> the target book.

An optimizer doing a balancing act:

    maximize   alpha' w           (expected return)
             - (lambda/2) w' V w  (penalty for risk, from the risk model)
             - kappa * cost(w-w0) (penalty for trading)
    subject to position limits, sector-neutrality, turnover cap, leverage.

What comes out is the **target portfolio**: the exact weight to hold in each
name. Constraints are not an afterthought — they encode the mandate (long-only?
market-neutral? how much can we turn over?) and keep the optimizer from chasing
estimation error into a degenerate corner.
"""

from alphaforge.portfolio.optimizer import OptimizerResult, optimize_portfolio

__all__ = ["optimize_portfolio", "OptimizerResult"]
