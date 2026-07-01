"""As-of joins — the KDB+ killer feature, via DuckDB.

An *as-of join* answers: "for each row on the left, give me the most recent row
on the right whose key was in effect *at or before* this timestamp." It is the
operation that is everywhere in finance and painfully slow in a normal DB:

  * attach the quote that was live at the instant of each trade,
  * attach the fundamentals that were *known as of* each rebalance date (no
    look-ahead!),
  * attach the security-master identifier valid on that date.

DuckDB exposes it as `ASOF JOIN ... ON left.t >= right.t`. KDB calls it `aj`.
Same idea, and it is the backbone of point-in-time correctness.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str | None = None,
    direction: str = "backward",
) -> pd.DataFrame:
    """As-of join `left` to `right` on a time column `on`, optionally within `by`.

    Parameters
    ----------
    on : the time/ordering column present in both frames.
    by : optional equality key joined first (e.g. ``security_id``) — the as-of
         match then happens *within* each group.
    direction : "backward" (default) takes the last right row at/<= the left
        time. This is the look-ahead-safe choice: you only ever see the past.

    Implemented with DuckDB's `ASOF JOIN` for correctness and speed; for small
    frames pandas' ``merge_asof`` would also work but DuckDB scales to the lake.
    """
    cmp = ">=" if direction == "backward" else "<="
    con = duckdb.connect()
    con.register("l", left)
    con.register("r", right)
    by_clause = f"l.{by} = r.{by} AND " if by else ""
    rcols = [c for c in right.columns if c != on and c != by]
    rsel = ", ".join(f"r.{c} AS {c}" for c in rcols)
    sep = ", " if rsel else ""
    q = f"""
        SELECT l.*{sep}{rsel}
        FROM l
        ASOF LEFT JOIN r
          ON {by_clause} l.{on} {cmp} r.{on}
    """
    out = con.sql(q).df()
    con.close()
    return out
