"""Data sources sit behind a single interface so they are swappable.

Today: `synthetic` (offline) and `yfinance` (free EOD equities).
Later: a `SnowflakeSource` / `S3DeltaSource` implementing the same `DataSource`
ABC — and nothing downstream (loaders, signals, risk, optimizer) changes. That
is the entire point of the abstraction: the JD's "Snowflake-backed pipelines /
AWS infra / many data sources, many endpoints" plugs in right here.
"""

from alphaforge.data.sources.base import DataSource, MarketData

__all__ = ["DataSource", "MarketData"]
