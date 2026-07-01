"""AlphaForge — an end-to-end multi-factor equity trading machine.

The package is laid out so that **each subpackage is one block of the
"multifactor machine" architecture** (see ARCHITECTURE.md):

    data/        (I)   DATA + SIGNALS      — ingestion, loaders, audit, universe
    refdata/     (IV)  SECURITY MATCHING   — point-in-time identifier mapping
    signals/           raw signals         — value/momentum/size/quality/...
    alpha/       (II)  ALPHA FORECASTS     — refine signals into tradable alphas
    risk/        (III) RISK MODEL          — factor covariance + specific risk
    portfolio/   (IV)  PORTFOLIO CONSTR.   — convex optimizer -> target book
    execution/   (V)   IMPL / TRADING      — cost model + fill simulator
    performance/ (VI)  PERFORMANCE         — attribution, IR, factor decay
    storage/     (VII) STORAGE             — Parquet/Delta warehouse + ASOF (KDB)
    backtest/          the engine that wires every block into one daily loop
"""

__version__ = "0.1.0"
