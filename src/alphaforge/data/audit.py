"""Data auditing — catch bad data as far upstream as possible.

"If the data's wrong, the factor trades on garbage." You cannot trust a vendor
to always ship correct data, so you wrap every feed in statistical checks and
alert when a metric crosses a threshold. The classic checks (straight from the
architecture):

  * **coverage %** — what fraction of the expected universe is present today?
    A sudden drop means the vendor shipped a truncated file.
  * **null / staleness** — columns that should be populated, that aren't.
  * **outliers** — values outside a sane range (e.g. |daily return| > 50%).
  * **row-count drift** — today's row count vs the trailing median.

Each check returns a pass/fail with the offending number, so an audit run reads
like a pre-flight checklist. In prod these emit to a dashboard + pager.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.logging import get_logger
from alphaforge.storage.warehouse import Warehouse

log = get_logger(__name__)


@dataclass
class AuditResult:
    check: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class AuditReport:
    table: str
    results: list[AuditResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, check: str, passed: bool, value: float, threshold: float, detail: str = "") -> None:
        self.results.append(AuditResult(check, passed, value, threshold, detail))


def audit_prices(wh: Warehouse, expected_names: int, max_abs_ret: float = 0.5) -> AuditReport:
    """Audit the prices table — the most important feed."""
    df = wh.read_table("prices", columns=["date", "security_id", "close", "ret", "volume"])
    rep = AuditReport("prices")

    # coverage: median names-per-day vs expected universe size
    per_day = df.groupby("date")["security_id"].nunique()
    coverage = float(per_day.median() / max(expected_names, 1))
    rep.add("coverage", coverage >= 0.95, coverage, 0.95,
            detail=f"median {per_day.median():.0f}/{expected_names} names/day")

    # nulls in close
    null_frac = float(df["close"].isna().mean())
    rep.add("null_close", null_frac <= 0.001, null_frac, 0.001)

    # non-positive prices (data error)
    bad_px = float((df["close"] <= 0).mean())
    rep.add("positive_prices", bad_px == 0.0, bad_px, 0.0)

    # return outliers
    out_frac = float((df["ret"].abs() > max_abs_ret).mean())
    rep.add("return_outliers", out_frac <= 0.0005, out_frac, 0.0005,
            detail=f"|ret|>{max_abs_ret:.0%}")

    # zero-volume staleness
    stale = float((df["volume"] <= 0).mean())
    rep.add("nonzero_volume", stale <= 0.02, stale, 0.02)

    _log_report(rep)
    return rep


def audit_coverage_match(coverage: float, threshold: float = 0.98) -> AuditResult:
    """Audit the security-matching coverage (fraction of vendor rows mapped)."""
    res = AuditResult("match_coverage", coverage >= threshold, coverage, threshold)
    log.info("audit.match_coverage", passed=res.passed, coverage=round(coverage, 4))
    return res


def _log_report(rep: AuditReport) -> None:
    for r in rep.results:
        level = log.info if r.passed else log.error
        level(
            "audit",
            table=rep.table,
            check=r.check,
            passed=r.passed,
            value=round(r.value, 5),
            threshold=r.threshold,
            detail=r.detail,
        )
