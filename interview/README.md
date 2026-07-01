# Interview PMS — the live-coding "portfolio management system"

This folder is the **round-1 interview problem** from the transcript, isolated:

> *"build a portfolio management system — something that could track orders,
> execute orders, and maintain positions across different stocks"* — with
> **stop losses, limit orders**, and P&L.

`portfolio_tracker.py` is a complete, ~200-line answer in pure standard-library
Python. This is the version you **code live**. (The big `alphaforge/` machine is
the version you **draw and discuss** in a system-design round — don't confuse
the two.)

```bash
python interview/portfolio_tracker.py        # run the demo scenario
pytest interview/test_portfolio_tracker.py   # 7 tests
```

## How to actually run this interview (communication > code)

The transcript's single biggest lesson: **half the grade is whether you can
explain your thinking clearly.** So don't start typing. Do this:

1. **Clarify + state assumptions (2 min).** "Are fills full or partial? Long-only
   or shorts too? Do we model commission/slippage? One price per symbol?" Then
   state what you'll assume. Interviewers *love* this — it's the difference
   between a junior and a senior.
2. **Sketch the design out loud (2 min).** "I'll have a `Portfolio` with cash, a
   `Position` per symbol, and a book of open orders. Market orders fill now;
   limit/stop orders rest and get triggered by an `on_price` tick. That tick is
   the matching engine."
3. **Build the happy path first.** Market buy → position + cash. Get *something*
   working before edge cases.
4. **Then layer complexity:** limit → stop → average-cost P&L → shorts.
5. **Talk while you code.** Narrate every decision ("I'll use average-cost
   accounting so I can realize P&L on partial exits").
6. **Test as you go.** Even one `assert` shows you think about correctness.

## The design in one breath (say this)

> "`Portfolio` owns cash, a `Position` per symbol (signed quantity + average
> price + realized P&L), and open orders. `submit_order` fills market orders
> immediately and rests limit/stop orders. `on_price(symbol, price)` is the
> engine: it updates the last price and triggers any resting order whose
> condition the new price satisfies. Fills use average-cost accounting — adding
> updates the average, reducing realizes P&L. Equity = cash + position value,
> and it always equals initial cash + realized + unrealized."

## Order-type logic (know this cold)

| Order | Fills when… | Typical use |
|---|---|---|
| **Market** | immediately, at current price | get in/out now |
| **Buy Limit** | price **≤** your limit | buy cheaper than now |
| **Sell Limit** | price **≥** your limit | take profit |
| **Buy Stop** | price **≥** stop | breakout entry |
| **Sell Stop** | price **≤** stop | **STOP-LOSS** (cut a loss) |

Memory hook: **limit = "or better"** (patient, price improvement); **stop =
"trigger then go"** (becomes a market order once crossed).

## Average-cost accounting (the P&L core)

- **Add to a position** → new avg = weighted average of old and new fills; no P&L.
- **Reduce/close** → realize `(fill_price − avg_price) × qty_closed × direction`.
- **Flip through zero** → realize on the closed part, re-open the remainder at the
  new price.

Invariant to state: **equity == initial_cash + realized_pnl + unrealized_pnl**
at all times. (There's a test for it.)

## Likely follow-up questions (and the one-line answer)

- **"Partial fills?"** → give `Order` a `filled_qty`; fill `min(remaining, available)`.
- **"Commissions/slippage?"** → subtract `commission + qty*slippage` in `_fill`
  and from cash.
- **"Multiple order books / an order-matching engine?"** → keep per-symbol sorted
  books (price-time priority) and match crossing orders — that's an exchange, a
  bigger question.
- **"Risk limits?"** → reject an order in `submit_order` if it would breach a
  position or gross-exposure cap.
- **"How would you make it fast for millions of orders?"** → per-symbol heaps for
  resting limit/stop levels so triggering is O(log n), not a full scan.
- **"How does this relate to the real system?"** → *this* is order/execution
  bookkeeping; the full research→trade machine (factors, risk model, optimizer)
  sits *above* it and produces the target positions this layer then executes.
  (That's your bridge to the `alphaforge/` project — mention it!)

## Why this is the perfect interview piece
It's small enough to finish, rich enough to show data-structure sense, OOP,
edge-case handling, and testing — and it's the *exact* problem quant desks ask.
Pair it with being able to explain the big machine, and you cover both interview
rounds: **the coding screen AND the system design.**
