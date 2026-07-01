"""Raw signals — individual, measurable views on a stock.

A *signal* is one number per name per day that expresses a view: "this name
looks cheap" (value), "this name has been winning" (momentum), etc. Signals are
NOT yet tradable — they're raw. The alpha layer (II) refines them. Classic
styles: value, momentum, size, quality, low-vol; plus exotic ones from alt-data.

Every signal is computed **cross-sectionally** (ranked against peers on the same
day) and made point-in-time safe (only uses information knowable as of date D).
"""

from alphaforge.signals.library import SIGNALS, compute_all_signals

__all__ = ["SIGNALS", "compute_all_signals"]
