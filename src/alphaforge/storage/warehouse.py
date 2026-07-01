"""Parquet + DuckDB warehouse — partitioned by date, queried with SQL.

Why this shape (straight from the architecture):

  * **Columnar (Parquet):** our queries touch a few columns across millions of
    rows ("give me close + volume for 2018"), not whole rows. Columnar storage
    means we read only the columns asked for, compress hard, and skip row-groups
    that can't match a filter.

  * **Partitioned by date:** the historical run is one big parallel scan; daily
    updates are a cheap append of a single new date partition.

  * **DuckDB as the query engine:** it reads Parquet directly, pushes filters
    down (predicate pushdown), and gives us `ASOF JOIN`. It's an in-process
    OLAP DB — the local analog of Snowflake/Databricks SQL.

`write_table` is *idempotent per partition*: re-running a date overwrites just
that date's file (the Delta-style "overwrite a partition" you need when a vendor
restates history) — never a half-written table.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alphaforge.logging import get_logger

log = get_logger(__name__)


class Warehouse:
    def __init__(self, root: str | Path = "data_lake") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths -------------------------------------------------------------
    def _table_dir(self, table: str) -> Path:
        d = self.root / table
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- writes ------------------------------------------------------------
    def write_table(self, table: str, df: pd.DataFrame, partition_col: str = "date") -> None:
        """Write a dataframe, one Parquet file per partition value (date).

        Hive-style layout: ``data_lake/<table>/<date>=YYYY-MM-DD/part.parquet``.
        Overwriting a single partition is how you cleanly absorb a vendor
        restatement without rebuilding the whole table.
        """
        if df.empty:
            return
        tdir = self._table_dir(table)
        # Normalize the partition key to a date string.
        keys = pd.to_datetime(df[partition_col]).dt.strftime("%Y-%m-%d")
        for key, chunk in df.assign(_k=keys).groupby("_k"):
            pdir = tdir / f"{partition_col}={key}"
            pdir.mkdir(parents=True, exist_ok=True)
            chunk.drop(columns="_k").to_parquet(pdir / "part.parquet", index=False)
        log.info("warehouse.write", table=table, rows=len(df), partitions=keys.nunique())

    def write_unpartitioned(self, table: str, df: pd.DataFrame) -> None:
        """Write a small reference table (e.g. the security master) as one file."""
        tdir = self._table_dir(table)
        df.to_parquet(tdir / "data.parquet", index=False)
        log.info("warehouse.write_ref", table=table, rows=len(df))

    # ---- reads -------------------------------------------------------------
    def _glob(self, table: str) -> str:
        tdir = self.root / table
        # Partitioned table?
        if any(tdir.glob(f"*={tdir.name}*")) or any(tdir.glob("*=*")):
            return str(tdir / "**" / "*.parquet")
        return str(tdir / "*.parquet")

    def read_table(
        self,
        table: str,
        columns: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        where: str | None = None,
    ) -> pd.DataFrame:
        """Read a table back with column- and date-pruning pushed into DuckDB.

        This is the "read only what you need" promise of a columnar lake: the
        `columns`/`start`/`end` args become predicate pushdown, so a 15-year
        table costs the same as the slice you actually touch.
        """
        pattern = self._glob(table)
        cols = ", ".join(columns) if columns else "*"
        # hive_partitioning=1 surfaces the `date=` folder as a real column.
        q = f"SELECT {cols} FROM read_parquet('{pattern}', hive_partitioning=1)"
        clauses: list[str] = []
        if start:
            clauses.append(f"date >= DATE '{start}'")
        if end:
            clauses.append(f"date <= DATE '{end}'")
        if where:
            clauses.append(where)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        return duckdb.sql(q).df()

    def sql(self, query: str) -> pd.DataFrame:
        """Escape hatch: run arbitrary DuckDB SQL (used by the ASOF helper)."""
        return duckdb.sql(query).df()

    def table_exists(self, table: str) -> bool:
        tdir = self.root / table
        return tdir.exists() and any(tdir.rglob("*.parquet"))
