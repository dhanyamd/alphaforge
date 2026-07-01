"""A portfolio management system — the classic quant live-coding question.

This is the round-1 interview problem from the transcript: "build something that
can track orders, execute orders, and maintain positions across different stocks"
with market / limit / stop orders and P&L. It is deliberately self-contained
(pure Python standard library) so you could reproduce it in ~an hour.

Design, stated out loud (interviewers grade communication as much as code):

  * A `Portfolio` holds cash + a `Position` per symbol + a book of open orders.
  * Orders come in three types:
      - MARKET : fill immediately at the current price.
      - LIMIT  : rest until the market reaches your price (buy<=limit, sell>=limit).
      - STOP   : rest until the market crosses a trigger, then fill at market
                 (buy stop = breakout entry; sell stop = STOP-LOSS).
  * `on_price(symbol, price)` is the market-data tick: it updates the last price
    and triggers any resting orders that should now fill. This is the engine loop.
  * Positions use average-cost accounting; selling above your average cost books
    realized P&L. Shorts (negative quantity) are handled symmetrically.

Assumptions I'd state to an interviewer (make them explicit!):
  * Fills are full (no partial fills) and instantaneous at the trigger price.
  * No commissions/slippage here (easy to add — see `extensions` in the README).
  * One price series per symbol; long and short both allowed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import count


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"          # stop-market: triggers, then fills at market


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    id: int
    symbol: str
    side: Side
    quantity: float
    type: OrderType
    limit_price: float | None = None   # for LIMIT
    stop_price: float | None = None    # for STOP
    status: OrderStatus = OrderStatus.OPEN
    fill_price: float | None = None

    def signed_qty(self) -> float:
        """+quantity for a buy, -quantity for a sell."""
        return self.quantity if self.side is Side.BUY else -self.quantity


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0     # signed: >0 long, <0 short
    avg_price: float = 0.0    # average entry price of the open position
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_price)


class Portfolio:
    def __init__(self, cash: float = 1_000_000.0) -> None:
        self.cash = cash
        self.initial_cash = cash
        self.positions: dict[str, Position] = {}
        self.open_orders: dict[int, Order] = {}
        self.order_history: list[Order] = []
        self.last_price: dict[str, float] = {}
        self._ids = count(1)

    # orders
    def submit_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Submit an order. MARKET fills now (if we have a price); others rest."""
        order = Order(next(self._ids), symbol, side, quantity, type, limit_price, stop_price)

        # Basic validation — reject nonsense early, and say why.
        if quantity <= 0:
            order.status = OrderStatus.REJECTED
        elif type is OrderType.LIMIT and limit_price is None:
            order.status = OrderStatus.REJECTED
        elif type is OrderType.STOP and stop_price is None:
            order.status = OrderStatus.REJECTED
        if order.status is OrderStatus.REJECTED:
            self.order_history.append(order)
            return order

        if type is OrderType.MARKET and symbol in self.last_price:
            self._fill(order, self.last_price[symbol])
        else:
            self.open_orders[order.id] = order
        return order

    def cancel_order(self, order_id: int) -> bool:
        order = self.open_orders.pop(order_id, None)
        if order is None:
            return False
        order.status = OrderStatus.CANCELLED
        self.order_history.append(order)
        return True

    # price feed
    def on_price(self, symbol: str, price: float) -> list[Order]:
        """A new market print for `symbol`. Update state + trigger resting orders.

        Returns the orders that filled on this tick. This is the matching engine:
        for each open order on this symbol, decide whether the new price triggers
        it, and if so, fill it.
        """
        self.last_price[symbol] = price
        filled: list[Order] = []
        for order in list(self.open_orders.values()):
            if order.symbol != symbol:
                continue
            if self._should_trigger(order, price):
                # LIMIT fills at its limit price (price improvement); MARKET/STOP at market.
                fill_px = order.limit_price if order.type is OrderType.LIMIT else price
                self._fill(order, fill_px)
                filled.append(order)
        return filled

    @staticmethod
    def _should_trigger(order: Order, price: float) -> bool:
        if order.type is OrderType.LIMIT:
            # Buy limit triggers when the market drops to/below your price; sell limit above.
            return price <= order.limit_price if order.side is Side.BUY else price >= order.limit_price
        if order.type is OrderType.STOP:
            # Buy stop (breakout) triggers when price rises to/through the stop;
            # sell stop (STOP-LOSS) triggers when price falls to/through the stop.
            return price >= order.stop_price if order.side is Side.BUY else price <= order.stop_price
        return True  # a resting MARKET order (submitted before any price) fills now

    #execution
    def _fill(self, order: Order, price: float) -> None:
        """Apply a full fill: update the position (avg-cost accounting) and cash."""
        pos = self.positions.setdefault(order.symbol, Position(order.symbol))
        q = order.signed_qty()

        if pos.quantity == 0 or (pos.quantity > 0) == (q > 0):
            # Opening or adding in the same direction -> update the average price.
            total = abs(pos.quantity) + abs(q)
            pos.avg_price = (abs(pos.quantity) * pos.avg_price + abs(q) * price) / total
            pos.quantity += q
        else:
            # Reducing / closing / flipping -> realize P&L on the closed portion.
            closed = min(abs(q), abs(pos.quantity))
            direction = 1.0 if pos.quantity > 0 else -1.0
            pos.realized_pnl += closed * (price - pos.avg_price) * direction
            pos.quantity += q
            if abs(q) > closed:          # flipped through zero -> new position at fill price
                pos.avg_price = price
            elif pos.quantity == 0:
                pos.avg_price = 0.0

        self.cash -= q * price           # buy spends cash; sell adds cash
        order.status = OrderStatus.FILLED
        order.fill_price = price
        self.open_orders.pop(order.id, None)
        self.order_history.append(order)

    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self.last_price.get(p.symbol, p.avg_price))
            for p in self.positions.values()
        )

    def equity(self) -> float:
        """Total account value = cash + market value of all positions.

        Invariant worth stating: equity == initial_cash + realized + unrealized.
        """
        mv = sum(p.market_value(self.last_price.get(s, p.avg_price)) for s, p in self.positions.items())
        return self.cash + mv

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = list(self.open_orders.values())
        return [o for o in orders if symbol is None or o.symbol == symbol]

    def summary(self) -> str:
        lines = [f"cash={self.cash:,.2f}  equity={self.equity():,.2f}  "
                 f"realized={self.realized_pnl():,.2f}  unrealized={self.unrealized_pnl():,.2f}"]
        for s, p in self.positions.items():
            if p.quantity != 0:
                px = self.last_price.get(s, p.avg_price)
                lines.append(f"  {s}: {p.quantity:+.0f} @ avg {p.avg_price:.2f} "
                             f"(last {px:.2f}, uPnL {p.unrealized_pnl(px):+,.2f})")
        return "\n".join(lines)


def _demo() -> None:
    """A scenario you can narrate to an interviewer, step by step."""
    pf = Portfolio(cash=100_000)

    print("1) Market buy 100 AAPL @ 150")
    pf.on_price("AAPL", 150.0)
    pf.submit_order("AAPL", Side.BUY, 100, OrderType.MARKET)
    print("   ", pf.summary())

    print("2) Rest a STOP-LOSS sell 100 AAPL @ 140, and a LIMIT sell 100 @ 170")
    pf.submit_order("AAPL", Side.SELL, 100, OrderType.STOP, stop_price=140.0)
    pf.submit_order("AAPL", Side.SELL, 100, OrderType.LIMIT, limit_price=170.0)
    print("   open orders:", [(o.id, o.type.value, o.side.value) for o in pf.get_open_orders()])

    print("3) Price rises to 172 -> the LIMIT sell fills at 170 (we take profit)")
    filled = pf.on_price("AAPL", 172.0)
    print("   filled:", [(o.id, o.type.value, o.fill_price) for o in filled])
    print("   ", pf.summary())
    print("   (the untriggered stop-loss should be cancelled since we're flat)")
    for o in pf.get_open_orders("AAPL"):
        pf.cancel_order(o.id)
    print("   ", pf.summary())


if __name__ == "__main__":
    _demo()
