# CONCEPTS — the multifactor machine, block by block

This is the teaching companion. For each block: the **idea**, the **math**, how
it's **implemented here**, and the **production reality**. Read top to bottom and
you'll understand how a real quant desk's pipeline fits together.

The blueprint is Grinold & Kahn, *Active Portfolio Management*. The one sentence
to internalize first:

> **Active management = forecasting the market's errors.** The price already
> bakes in a consensus; our job is to find the specific spots where we *disagree*
> and be right more often than the others trying to do the same.

And the two numbers that tie the whole machine together:
- **Information Ratio:** `IR = active_return / active_risk` — the report card.
- **Fundamental Law:** `IR ≈ IC · √breadth` — two ways to improve: more **skill**
  per bet (IC), or more **independent** bets (breadth). The word *independent* is
  doing enormous work — 5 long retail + 5 short energy is **2** bets, not 10.

---

## (VII) STORAGE — `storage/`
**Idea.** Two tools for two jobs, exactly as real firms run it.
- **Warehouse** = Parquet + DuckDB: columnar, cheap, scales out, remembers all
  history, partitioned by date. (Our local stand-in for S3 + Delta/Snowflake.)
- **Engine** = the as-of join: "give me the value in effect *as of* this time" —
  the one operation KDB+ is famous for. DuckDB ships it as `ASOF JOIN`.

**Why columnar + partition-by-date.** Our queries touch a few columns across
millions of rows ("close+volume for 2018"), not whole rows. Columnar storage
reads only the columns asked for, compresses hard, and skips row-groups that
can't match a filter. Daily updates are a cheap append of one date partition; a
15-year historical run is one big parallel scan.

**Why ASOF matters.** Joining two time series is hard because timestamps never
line up. ASOF joins on inequality (`left.t >= right.t`) to grab the closest
*preceding* row — the look-ahead-safe choice. It is everywhere in finance: attach
the live quote at each trade, the known fundamentals at each rebalance, the valid
identifier on each date.

**Files.** `warehouse.py` (write/read partitioned Parquet via DuckDB),
`asof.py` (`asof_join` wrapper).

---

## (IV) SECURITY MATCHING — `refdata/`
**Idea.** Vendors name the same company differently (CUSIP/ISIN/SEDOL/ticker/URL)
and those IDs **change over time** (M&A, ticker changes). The security master
reconciles them, stamping every mapping with a `[valid_from, valid_to)` window.

**Point-in-time is a hard requirement.** An ID is only valid for a date range.
Ask "what was true *as of that date*", never "what is true today". Get it wrong
and you inject **look-ahead bias** that silently inflates every backtest.

**Implementation.** `build_security_master` synthesizes a crosswalk with realistic
mid-history ID-change events (~15% of names). `SecurityMaster.match` resolves a
vendor key + date to the internal id via an ASOF join, then nulls out rows that
fell past a window's end. `coverage()` = fraction mapped — the exact metric the
data audit watches.

**Production reality.** You'd seed this from CUSIP Global Services / OpenFIGI
(OpenFIGI is free). Terabyte-scale datasets get matched with Spark/Polars.

---

## (I) DATA + SIGNALS — `data/`, `signals/`
**Idea.** Everything downstream is only as good as the data. This block is the
unglamorous, decisive work.

**Loaders** (`data/loaders.py`). One per vendor (formats/cadence differ), all
ending the same way: write a clean, point-in-time, date-partitioned table so any
factor built on top re-runs automatically on fresh data. A broken loader **halts**
the factor — so loaders are production-critical with auditing wrapped around them.

**Data audit** (`data/audit.py`). "If the data's wrong, the factor trades on
garbage — catch it as far upstream as possible." Checks: **coverage %** (truncated
vendor file?), **nulls/staleness**, **outliers** (`|ret|>50%`), **non-positive
prices**, **zero-volume**. Each returns pass/fail vs a threshold; in prod these
page someone.

**Universe** (`data/universe.py`). *Define your universe first.* Trading
everything maximizes breadth but illiquid names cost more to trade than they
add. We gate on trailing **ADV** (average dollar volume) and keep the top-N most
liquid names, point-in-time (a name enters/leaves as its liquidity changes).

**Signals** (`signals/`). One number per name per day expressing a view, built
only from information knowable as of D:
- **momentum** = 12-1 month return (skip the last month to avoid short-term
  reversal). **size** = −log(mktcap). **low_vol** = −trailing 63d vol.
- **value** = book-to-price, **quality** = gross profitability — attached
  **as-of** from fundamentals with the reporting lag respected.
- **altdata** = a weekly alt-data score, attached as-of.

**Signal hygiene** (`signals/base.py`). Per cross-section: **winsorize** (clip
outliers) → **neutralize** (regress out sector & size, keep the residual — this
is what makes a signal an *independent* bet, not a disguised sector tilt) →
**z-score** (put every signal on the same scale so they can be combined).

---

## (II) ALPHA FORECASTS — `alpha/`
**Idea.** A raw signal is just a ranking with no units. An **alpha** is a refined
forecast of *residual return* the optimizer can trade. Grinold's refinement:

```
        alpha = volatility × IC × score
```
- **volatility** — how much the name can move (the opportunity size),
- **IC** (information coefficient) — the signal's genuine predictive skill
  (cross-sectional rank-corr of signal vs forward return; real equity ICs are
  tiny: 0.02–0.05),
- **score** — the cleaned, standardized signal for this name now.

**Multi-factor** = run many alphas at once, each ideally an independent edge,
then blend (here: IC-weighted, so more-skillful signals get more say). The
blended `alpha` is the optimizer's expected-return input.

**Files.** `refine.py` (clean panel, forecast vol, `empirical_ic`, `refine`),
`combine.py` (IC-weighted blend → `alpha`). The **research→production pod** lives
conceptually here: a researcher writes the methodology, a developer productionizes
it (vectorized, distributed), it gets a 5–15yr historical run, then a live daily
run — and because **factors decay**, the loop never ends.

---

## (III) RISK MODEL — `risk/`
**Idea (the block newcomers underrate).** Alpha says what you hope to make; the
risk model says what it could cost in volatility. The key trick that makes the
math computable:

```
   r_i = Σ_k B_{i,k} f_k + u_i          (return = factor exposures + specific)
   V    = B Σ_f B' + D                  (asset covariance)
```
For 1,400 names a full stock-by-stock covariance needs ~980,000 pairwise numbers
— hopeless to estimate from finite history (you'd fit noise). Describe each stock
as exposures to ~65 common factors and you need only ~2,000 numbers (the factor
covariance). An impossible problem becomes tractable.

**Estimation (Barra-style).**
- `factor_model.py` — each day, **cross-sectional WLS regression** of realized
  returns on known exposures (sector dummies + style z-scores), weighted by
  √mktcap. Fitted coefficients = **factor returns** `f_t`; residuals = **specific
  returns** `u_{i,t}`.
- `covariance.py` — **Σ_f** = EWMA of factor returns (half-life ~90d) + diagonal
  **shrinkage** (Ledoit-Wolf style, ~0.2) to tame sampling error. **D** = per-name
  EWMA of squared specific returns (half-life ~60d), shrunk toward the
  cross-sectional median.
- `model.py` — `RiskModel.fit` builds the full history once; `snapshot(date)`
  serves the point-in-time `(B, Σ_f, D)`.

**Two intuitions baked in.** (1) **Diversification:** portfolio risk is *less*
than the weighted average of its parts because names don't all move together —
but only the *specific* part averages away; **systematic** (market) risk never
diversifies (CAPM/APT). (2) **√-time:** risk doesn't add across time but variance
does, so risk grows with √time — annualize daily vol by ×√252, not ×252
(`annualize_risk`).

---

## (IV) PORTFOLIO CONSTRUCTION — `portfolio/`
**Idea.** An optimizer balancing act:
```
   maximize  alpha'w  −  (λ/2) w'Vw  −  κ·cost(w−w₀)
   s.t.  position limits · sector-neutral · turnover cap · leverage · long-only?
```
Output = the **target portfolio**: the exact weight per name.

**Implementation.** `optimizer.py` writes this as a convex QP in **CVXPY** and
hands it to OSQP. Constraints encode the *mandate*: long-only & fully-invested, or
dollar-neutral long-short; per-name `|w|≤max_weight`; gross `Σ|w|≤max_gross`;
**sector-neutral** (no active sector bet vs an equal-weight benchmark); **turnover**
`Σ|w−w₀|≤max_turnover` (exempted on the initial build — see Debug Log). A
risk-scaled analytic fallback (`w ∝ V⁻¹α`, clipped) keeps a long run alive if a
solver hiccups. Constraints aren't an afterthought — they stop the optimizer
chasing estimation error into a degenerate corner.

---

## (V) IMPLEMENTATION + TRADING — `execution/`
**Idea.** The optimizer hands you a target; getting there leaks alpha. The
philosophy: **subtract as little value as possible.** Four costs:
- **commission** (per-share broker fee), **spread** (buy at ask, sell at bid),
- **market impact** — the big sneaky one: buying size pushes the price against
  you. "You can't observe the market without disturbing it." It follows the
  **square-root law:** `impact ≈ η·σ·√(Q/ADV)` — trading a small slice is nearly
  free; cost grows with the *√* of participation (which is why you slice big
  orders over time).
- **opportunity cost** — the fill you waited on that ran away.

**Implementation.** `costs.py` (`linear_cost_coef` = convex commission+spread for
the optimizer; `realized_cost` = full nonlinear impact for the simulator).
`simulator.py` fills toward target under a **participation cap** (≤10% of ADV);
unfilled remainder carries over (a source of opportunity cost). `shortfall.py`
measures **implementation shortfall** = frictionless paper book vs real book.

---

## (VI) PERFORMANCE — `performance/`
**Idea.** After the fact, separate **skill from luck** and find where the skill
lives, so you double down. This block feeds straight back to research.

- `metrics.py` — Sharpe, **Information Ratio**, drawdown, hit rate, and the
  **fundamental-law** decomposition `IR ≈ IC·√breadth`.
- `attribution.py` — split each period's P&L into **factor** (the bets you
  intended: value/momentum/sectors) vs **specific** (name-selection skill/noise),
  using the same factor decomposition as the risk model.
- `decay.py` — track each factor's **rolling IC** through time. A factor whose IC
  trends to zero is **decaying** (crowded out as others discover it) — the signal
  that triggers a second research project. *Factors decay → the research loop
  never ends.*

---

## A crucial honesty note (read this)
In this synthetic build the realized IR (~2.7) sits **above** the fundamental-law
IR (~1.3). That gap is a *feature to understand*, not a bug: the alpha here uses
the **true** factor ICs (we configured them) as if known perfectly, and there are
no costs on the paper book. In the real world you must **estimate** IC, it's
unstable out-of-sample, and that instability roughly halves live IR. The single
biggest reason live results disappoint vs backtests is exactly this — in-sample,
known-skill assumptions. The machine is built so you can *see* that gap.
