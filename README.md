# AlphaForge — Production-Grade Multi-Factor Equity Backtesting Engine

> A from-scratch, runnable implementation of the industry-standard production quant architecture — **"The Multifactor Machine"** — with every block of the pipeline built and wired into a single backtest. Designed to mirror how a quant research engineer's work fits together in a real-world trading system.

It runs **fully offline** on a synthetic market with embedded factor structure (so your signals have genuine predictive power). Every storage boundary is abstracted behind interfaces that can later be pointed at cloud services like **Snowflake, AWS S3, MotherDuck, or Databricks** (see [`docs/services_free_vs_paid.md`](docs/services_free_vs_paid.md)).

---

## 1. System Architecture

Below is the chronological flow of data through the engine. It details how raw records on your hard drive become standardized signals, are optimized with risk constraints, executed under volume caps, and finally evaluated for performance and signal decay.

```mermaid
graph TD
    %% Define Nodes and Styles
    subgraph Data & Storage Layer
        A["Prices & Fundamentals (Parquet on Disk)"] -->|DuckDB Query Engine| B["signals/library.py: compute_all_signals"]
        A -->|Hive-Partitioned Lake| C["data/universe.py: build_universe"]
    end
    
    subgraph Signal Hygiene & Alpha Generation
        B -->|Raw Signals| D["alpha/refine.py: clean_panel"]
        C -->|Universe Selection| D
        D -->|winsorize, neutralize, z-score| E["alpha/combine.py: clean_signal_panel"]
        E -->|Cleaned Features| F{"Alpha Model Selector"}
        F -->|Linear blend| G["build_alphas (Linear prior IC weighting)"]
        F -->|Non-linear GBT| H["build_ml_alphas (Walk-forward GBT)"]
    end
    
    subgraph Portfolio Construction & Optimization
        G -->|Expected Returns| I["portfolio/optimizer.py: optimize_portfolio"]
        H -->|Expected Returns| I
        J["risk/model.py: RiskModel"] -->|Covariance & Factor Exposures| I
        I -->|CVXPY Convex QP Solver| K["Target Weights (Uncapped)"]
    end
    
    subgraph Execution & Trading Simulation
        K --> L["execution/simulator.py: simulate_fills"]
        L -->|Enforce ADV Participation Cap| M["Realized Weights (Real Book)"]
        L -->|Instant Frictionless Fill| N["Target Weights (Paper Book)"]
        L -->|Commissions + Slippage (Slippage: Square-root Impact)| O["Transaction Cost (pending_cost)"]
    end
    
    subgraph Daily P&L Loop
        M --> P["accrue P&L (yesterday's weights * today's returns)"]
        O -->|Subtracted from Real return tomorrow| P
        P -->|drift positions with price movement| Q["Drift Weights (_drift)"]
        Q -->|Next day's holding| M
    end
    
    subgraph Performance & Attribution
        P --> R["performance/decay.py: rolling_ic & decay_report"]
        P --> S["performance/metrics.py: tearsheet generation"]
    end

    classDef box fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#f3f4f6;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S box;
```

---

## 2. Package Structure & Key Modules

Each architecture block maps directly to a Python subpackage inside `src/alphaforge/`:

| Block | Package / File | Description |
|---|---|---|
| **(I) Data & Signals** | `data/universe.py`<br>`signals/library.py` | Trailing Average Daily Volume (ADV) universe selector. Computes raw signals: Momentum (12-1 month), Size (-log market cap), Low Volatility, Value (Book-to-Price), Quality, and Alt-Data. |
| **(II) Alpha Forecasts** | `alpha/refine.py`<br>`alpha/combine.py`<br>`alpha/ml_model.py` | Applies Winsorizing, sector/size neutralization, and standardizes z-scores. Blends signals using either a linear prior IC weighting (Grinold formula: `α = vol · IC · score`) or a Walk-Forward Gradient Boosted Tree (GBT) model. |
| **(III) Risk Model** | `risk/model.py` | Barra-style cross-sectional factor risk model estimating factor exposures and portfolio covariance matrix $V = B\Sigma_f B^T + D$. |
| **(IV) Portfolio Construction** | `portfolio/optimizer.py` | Formulates and solves a Convex Quadratic Program (QP) using `CVXPY`. Enforces max position weights, gross leverage limits, sector neutrality, and turnover caps. |
| **(V) Execution / Trading** | `execution/simulator.py`<br>`execution/costs.py` | Reality check simulation. Enforces a volume participation cap (e.g., max 5% of ADV) and calculates transaction costs (Linear commissions + non-linear square-root market impact slippage). |
| **(VI) Performance** | `performance/metrics.py`<br>`performance/decay.py` | Generates tear-sheets (Sharpe, Information Ratio, Drawdown), checks Fundamental Law metrics ($IR \approx IC \cdot \sqrt{breadth}$), and tracks factor IC decay over time to flag underperforming signals. |
| **(VII) Storage** | `storage/warehouse.py` | High-performance columnar data warehouse using **Parquet files** on disk and the **DuckDB** engine to execute predicate-pushdown queries and ASOF joins. |
| **(VIII) Distributed Twin** | `spark/` | A PySpark and Delta Lake implementation of the data pipelines, enabling identical logic to run over terabytes on Databricks clusters. |
| **Engine** | `backtest/engine.py`<br>`backtest/pipeline.py` | Wires the entire chronological day-by-day loop together, tracking the Real, Paper, and Benchmark portfolios while managing weight drift. |

---

## 3. Financial & Machine Learning Core Concepts

### A. Non-Linear Market Impact & Trading Costs
Trading is not free. The engine simulates realistic execution costs using Almgren's Square-Root Law:
$$\text{Impact Fraction} = \text{Impact Coefficient} \times \text{Daily Volatility} \times \sqrt{\frac{\text{Trade Dollars}}{\text{Average Daily Volume (ADV) Dollars}}}$$
*   **The Optimizer** uses a convex linear cost coefficient proxy (commission + half-spread) to remain fast and solvable.
*   **The Simulator** applies the full non-linear market impact above. If a trade is too large, it caps execution at the configured **Participation Cap** (e.g. 5% of ADV) to prevent ruinous slippage.

### B. Machine Learning (GBT) Setup
When GBT mode is enabled in configuration, the system switches from linear weighting to ML:
*   **Features (X):** Cleaned, winsorized, and sector-neutralized signal z-scores (e.g., Value and Momentum).
*   **Target (y):** Next-day forward returns **minus the cross-sectional mean of that day** (demeaning). This forces the model to predict relative outperformance (alpha) rather than guessing market direction.
*   **Time-Series Firewall:** Enforced via a **walk-forward validation loop** which trains models strictly on expanding windows of past data, ensuring no future data leaks into the predictor.

---

## 4. Quickstart Guide

### Installation
Deploy the local Python virtual environment (requires Python 3.12, uses `uv` if present, otherwise falls back to `pip`):
```bash
# Set up virtual environment
uv venv --python 3.12 .venv && source .venv/bin/activate

# Install package with dependencies
uv pip install -e ".[dev,viz]"
```

### Run the Pipeline Command-Line Tools
AlphaForge includes a command-line interface to execute each step:

```bash
# 1. Ingest synthetic data (generates prices, master reference data, and fundamentals)
alphaforge generate-data

# 2. Audit the database for coverage, outliers, missing values, and data gaps
alphaforge audit

# 3. Run the backtest (simulates trading across history and saves performance logs)
alphaforge backtest

# 4. View performance summary tables (reprints the tearsheet)
alphaforge report
```

### Advanced Diagnostics
```bash
# Compare models: Compare GBT (ML) vs. Linear weights out-of-sample (Sharpe, IC, Spread)
alphaforge compare-models

# Feature Importance: Run permutation importance on GBT to see which factors carry the edge
alphaforge feature-importance
```

### Switch to Real Market Data
You can switch the simulation to run on real equities:
```bash
uv pip install -e ".[realdata]"
# Edit config/backtest.yaml to set 'data.source: "yfinance"'
alphaforge generate-data && alphaforge backtest
```

---

## 5. What "Good" Looks Like
Running the backtest generates a performance tearsheet tracking:
*   **Annualized Return & Volatility:** The net profit and risk profile.
*   **Sharpe & Information Ratio (IR):** The consistency of risk-adjusted excess returns.
*   **Max Drawdown:** The maximum historical peak-to-trough drop.
*   **Implementation Shortfall:** The exact percentage of returns lost to execution commissions and slippage.
*   **Factor IC Decay:** A status report flagging whether your individual signals are `"stable"` or `"DECAYING"` over time.
