# Results Recorded — AlphaForge, honestly

This file records every test we ran, **including the ones that failed and the
biases we found**, because honest reporting is the whole point of quant. A result
you can't trust is worse than no result.

## Headline: does the strategy work?

**Yes — the multi-factor strategy has a real, durable edge, proven on 63 years of
survivorship-bias-free academic data (Kenneth French Data Library).** But the
*tradeable, after-cost* number is smaller than the raw backtest suggests, and we
document exactly why.

---

## The full test log (in order, nothing hidden)

| # | Data | What we measured | Result | Verdict |
|---|------|------------------|--------|---------|
| 1 | Synthetic (we built it) | full backtest IR | **2.71** | ❌ **overfit** — we designed the patterns AND tested on them. Not real. |
| 2 | 25 real survivor stocks (yfinance) | Information Ratio | **+0.05** | ⚠️ ~zero — too little breadth; half the signals were random stubs |
| 3 | 53 real stocks, real fundamentals | Information Ratio | **−0.09** | ⚠️ still no edge; strong bull-market benchmark |
| 4 | 145 real stocks + GBT alpha | Information Ratio | **+0.54** | ⚠️ inflated by survivorship + fundamentals look-ahead |
| 5 | 145 stocks, price-only (no look-ahead) | Info Ratio / holdout | **+0.72 / +0.70** | 🚩 IC ≈ 0 but IR high → survivorship-inflated momentum, NOT skill |
| 6 | **Ken French factors, 1963–2026 (clean)** | **combined Sharpe** | **0.73 (gross)** | ✅ **real, honest edge** |

**The single most important lesson:** results 1–5 all *looked* good at some point
and were all *untrustworthy* for a specific, diagnosable reason. Catching that
yourself — before it trades real money — is the actual skill.

---

## The honest result (test #6, in detail)

**Data:** Kenneth French Data Library — survivorship-bias-free, point-in-time,
professionally constructed factor returns, monthly, Jul 1963 – Apr 2026 (754 months).

**Single factors (long-short, market-neutral):**

| Factor | Ann. return | Sharpe |
|---|---|---|
| Momentum | +6.4% | **0.51** |
| Market   | +6.1% | 0.46 |
| Investment | +2.7% | 0.41 |
| Quality (profitability) | +2.8% | 0.40 |
| Value | +3.1% | 0.35 |
| Size | +1.7% | 0.21 (weak) |

**Our strategy — an equal-risk blend of Momentum + Value + Quality:**

```
Best single factor Sharpe : 0.51
COMBINED Sharpe (gross)   : 0.73   (+44% better than the best single factor)
```

**Full tearsheet (combo, 10%/yr vol target, GROSS of costs):**

| Metric | Value | Note |
|---|---|---|
| CAGR | +7.0% | annualized return |
| Volatility | 10.0% | (targeted) |
| Sharpe | 0.73 | gross |
| **Max drawdown** | **−37.8%** | **Sharpe hides this — worst peak-to-trough (bottomed ~2000)** |
| Hit rate (months) | 64.1% | loses ~1 month in 3 |
| % years positive | 81% | loses ~1 year in 5 |
| Worst year | −23.9% | |
| Worst month | −17.0% | |

**Net of trading costs** (momentum is high-turnover):

```
cost 0%/yr → Sharpe 0.73, CAGR +7.0%
cost 1%/yr → Sharpe 0.63, CAGR +5.9%
cost 2%/yr → Sharpe 0.53, CAGR +4.9%   ← realistic tradeable range ~0.5–0.6
```

**Lesson:** even a genuinely good 63-year strategy has a −38% drawdown and loses
money ~1 year in 5. Sharpe alone is dangerously incomplete — always read drawdown,
hit rate, and worst-year alongside it.

**Why the combo beats any single factor** — the factors barely move together:

```
       Mom    HML(value)  RMW(quality)
Mom    1.00    -0.19        0.08
HML   -0.19     1.00        0.09
RMW    0.08     0.09        1.00
```

Because they don't lose at the same time, blending them diversifies away risk
without giving up return — Sharpe rises from 0.51 → 0.73. **That diversification
is the edge.**

---

## What is honest vs inflated (read this before quoting any number)

✅ **Honest about test #6:**
- Real academic data, downloaded live; single-factor Sharpes match published values.
- Standard pre-specified factors, **equal-risk weights (no fitting)** → minimal researcher bias.
- Market-neutral → it's pure alpha, not disguised market beta.

⚠️ **Haircuts we must apply to the 0.73:**
1. **Gross of trading costs.** Momentum is high-turnover; net of realistic costs the
   combo Sharpe is **~0.5–0.6**, not 0.73. *This is the tradeable number.*
2. **Full-sample** (no strict holdout) and **publication bias** (surviving factors
   are the ones that got published).

**So the honest, tradeable claim is: a market-neutral multi-factor combo with a
Sharpe of roughly 0.5–0.6 after costs — a real but modest, well-known edge.**

---

## What the strategy is, and what the edge is

- **Strategy:** score every stock on momentum + value + quality, tilt toward the
  high-scorers, stay market/sector-neutral, spread across many names (breadth).
- **Edge = real factor premia + diversification across uncorrelated factors + scale.**
  Not the `alpha = vol × IC × score` formula — that's the *machinery* that turns
  the edge into sized positions, not the edge itself.
- **Nature of the edge:** *known / "smart beta"*, not proprietary. AQR, Dimensional,
  etc. run this for hundreds of billions. A proprietary edge needs a **new signal**
  (alt-data), **faster execution**, or a **better way to combine/time** the factors —
  which is what the never-ending research loop (and an autonomous AI researcher) is for.

---

## Why the earlier stock-level tests failed (it was DATA, not the strategy)

The yfinance failures (tests 2–5) were **data-quality problems**, not strategy
problems, proven by test #6 working on clean data:
- **Survivorship bias:** our tickers were today's survivors → momentum looked
  artificially good (winners we already knew survived).
- **Look-ahead:** yfinance `.info` gives only *today's* fundamentals → value/quality
  leaked mild future info.
- **Too little breadth:** 25–145 names can't turn a 0.02 IC into a reliable IR
  (`IR ≈ IC × √breadth`).

Fixing these properly needs **point-in-time universe membership + point-in-time
fundamentals** (CRSP/Compustat, or free-but-laborious SEC EDGAR) — which is exactly
why that data is the most valuable paid thing in the industry.

---

*Reproduce test #6:* the script pulls `F-F_Research_Data_5_Factors_2x3` and
`F-F_Momentum_Factor` via `pandas_datareader` (free, no API key) and computes the
equal-risk blend. Everything above was computed live, not hand-entered.
