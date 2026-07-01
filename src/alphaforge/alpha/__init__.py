"""(II) ALPHA FORECASTS — turn raw signals into tradable return forecasts.

A raw signal is just a ranking; it has no units. An *alpha* is a clean, refined
forecast of residual return — something the optimizer can actually trade off
against risk and cost. The refinement is Grinold's formula:

        alpha = volatility x IC x score

  * **volatility** — how much the stock can move (the opportunity size),
  * **IC** (information coefficient) — the signal's genuine predictive skill
    (cross-sectional corr between signal and forward return),
  * **score** — the standardized signal value for this name right now.

Each refined signal becomes one factor; running many at once is "multi-factor".
"""

from alphaforge.alpha.combine import build_alphas

__all__ = ["build_alphas"]
