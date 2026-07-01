"""Per-vendor data loaders — pull -> transform/aggregate -> write to the lake.

In production every vendor gets its *own* loader (their formats, cadence, and
quirks differ), but they all end the same way: write a clean, point-in-time,
date-partitioned table into the internal warehouse so that any factor built on
top runs again automatically on fresh data. A broken loader halts the factor —
so loaders are production-critical code with auditing wrapped around them.

Here we route a `MarketData` bundle into four warehouse tables. The
`security_master` is built and stored so the matching step (IV) can run.
"""
from __future__ import annotations

import pandas as pd

from alphaforge.config import Config
from alphaforge.data.sources.base import DataSource, MarketData
from alphaforge.data.sources.synthetic import SyntheticSource
from alphaforge.logging import get_logger
from alphaforge.refdata.security_master import build_security_master
from alphaforge.storage.warehouse import Warehouse

log = get_logger(__name__)


def make_source(cfg: Config) -> DataSource:
    """Factory: pick the data source from config (swap point for Snowflake/S3)."""
    if cfg.data.source == "synthetic":
        return SyntheticSource(cfg)
    if cfg.data.source == "yfinance":
        from alphaforge.data.sources.yfinance_source import YFinanceSource
        return YFinanceSource()
    raise ValueError(f"unknown data source: {cfg.data.source}")


def ingest(cfg: Config, wh: Warehouse) -> MarketData:
    """Full ingestion: fetch from the source, then load each vendor table.

    Returns the in-memory bundle too (handy for tests / notebooks), but the
    durable artifact is the warehouse — everything downstream reads from there.
    """
    source = make_source(cfg)
    md = source.fetch(cfg.run.start_date, cfg.run.end_date)

    # ---- vendor: prices ---------------------------------------------------
    _load_prices(wh, md.prices)
    # ---- vendor: fundamentals --------------------------------------------
    _load_fundamentals(wh, md.fundamentals)
    # ---- vendor: altdata --------------------------------------------------
    _load_altdata(wh, md.altdata)
    # ---- reference: static + point-in-time security master ----------------
    wh.write_unpartitioned("security_static", md.static)
    sec_master = build_security_master(
        md.security_ids(), cfg.run.start_date, cfg.run.end_date, seed=cfg.run.seed
    )
    wh.write_unpartitioned("security_master", sec_master)
    log.info("ingest.done", vendors=cfg.data.vendors, names=len(md.security_ids()))
    return md


def _load_prices(wh: Warehouse, df: pd.DataFrame) -> None:
    """Transform: ensure types, derive dollar-volume (ADV input), partition by date."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dollar_volume"] = out["close"] * out["volume"]
    out["mktcap"] = out["close"] * out["shares_out"]
    wh.write_table("prices", out, partition_col="date")


def _load_fundamentals(wh: Warehouse, df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Derived field at load time: gross profitability (Novy-Marx quality proxy).
    out["gross_profitability"] = out["gross_profit"] / out["total_assets"].replace(0, pd.NA)
    wh.write_table("fundamentals", out, partition_col="date")


def _load_altdata(wh: Warehouse, df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    wh.write_table("altdata", out, partition_col="date")
