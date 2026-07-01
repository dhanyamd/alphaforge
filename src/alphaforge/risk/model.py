"""RiskModel — assembles exposures, factor covariance, and specific risk.

Usage in the backtest:
    rm = RiskModel(cfg).fit(alpha_panel, prices, static)
    snap = rm.snapshot(date, universe_ids)      # point-in-time B, F, d
    var  = snap.portfolio_variance(weights)     # w' (B F B' + D) w

We estimate factor returns ONCE over the whole sample (each day's cross-sectional
regression uses only that day's data, so there's no look-ahead), then at each
rebalance we form the covariance from the trailing `lookback` window — i.e. only
history available as of that date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.config import Config
from alphaforge.logging import get_logger
from alphaforge.risk.covariance import factor_covariance, specific_variance
from alphaforge.risk.factor_model import build_exposures, estimate_factor_returns

log = get_logger(__name__)

# Risk does not add across time, but variance does (if returns are ~uncorrelated
# day to day). So to annualize a daily risk number you scale variance by the
# number of periods and take the square root: risk grows with sqrt(time).
TRADING_DAYS = 252


def annualize_risk(daily_vol: float, periods: int = TRADING_DAYS) -> float:
    """Daily volatility -> annualized volatility via the square-root-of-time rule."""
    return float(daily_vol * np.sqrt(periods))


@dataclass
class RiskModelSnapshot:
    """A point-in-time risk model for one rebalance date."""

    date: pd.Timestamp
    security_ids: list[str]
    factor_names: list[str]
    B: np.ndarray          # exposures: (N names, K factors)
    F: np.ndarray          # factor covariance: (K, K)
    d: np.ndarray          # specific variances: (N,)

    def covariance(self) -> np.ndarray:
        """Full asset covariance V = B F B' + diag(d). Used by the optimizer."""
        return self.B @ self.F @ self.B.T + np.diag(self.d)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """w' V w — decomposed so you can read factor vs specific contribution."""
        factor_var = float(w @ self.B @ self.F @ self.B.T @ w)
        specific_var = float((w**2) @ self.d)
        return factor_var + specific_var

    def risk_decomposition(self, w: np.ndarray) -> dict[str, float]:
        factor_var = float(w @ self.B @ self.F @ self.B.T @ w)
        specific_var = float((w**2) @ self.d)
        total = factor_var + specific_var
        return {
            "total_vol_daily": float(np.sqrt(max(total, 0))),
            "total_vol_annual": annualize_risk(np.sqrt(max(total, 0))),
            "factor_frac": factor_var / total if total > 0 else 0.0,
            "specific_frac": specific_var / total if total > 0 else 0.0,
        }


class RiskModel:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.factor_returns: pd.DataFrame | None = None
        self.specific_returns: pd.DataFrame | None = None
        self.exposures: pd.DataFrame | None = None
        self.factor_names: list[str] = []

    def fit(self, alpha_panel: pd.DataFrame, prices: pd.DataFrame, static: pd.DataFrame) -> "RiskModel":
        """Estimate the full factor-return / specific-return history."""
        style_cols = [s.name for s in self.cfg.enabled_signals]
        expo, factor_names = build_exposures(alpha_panel, static, style_cols)

        # Align exposures at t-1 to the return realized at t (1-day forward).
        px = prices.sort_values(["security_id", "date"]).copy()
        px["date"] = pd.to_datetime(px["date"])
        px["fwd_ret"] = px.groupby("security_id")["ret"].shift(-1)
        ret = px[["date", "security_id", "fwd_ret"]]
        weights = px[["date", "security_id", "mktcap"]].rename(columns={"mktcap": "weight"})

        fr, sr = estimate_factor_returns(expo, ret, factor_names, weights)
        self.factor_returns, self.specific_returns = fr, sr
        self.exposures, self.factor_names = expo, factor_names
        log.info("risk.fit", days=len(fr), factors=len(factor_names))
        return self

    def snapshot(self, date: pd.Timestamp, universe_ids: list[str]) -> RiskModelSnapshot:
        """Build the point-in-time (B, F, d) for `date` over `universe_ids`."""
        assert self.factor_returns is not None and self.exposures is not None
        date = pd.Timestamp(date)
        lookback = self.cfg.risk.lookback

        # Factor covariance from the trailing window (only past data).
        fr_hist = self.factor_returns.loc[:date].tail(lookback)
        F = factor_covariance(fr_hist, self.cfg.risk.halflife_cov, self.cfg.risk.shrinkage)

        # Specific variances from the trailing window.
        sr = self.specific_returns
        sr_hist = sr[(sr["date"] <= date) & (sr["date"] > date - pd.Timedelta(days=lookback * 2))]
        d_series = specific_variance(sr_hist, self.cfg.risk.halflife_var, self.cfg.risk.shrinkage)

        # Exposures as of `date` (latest available on/before the date per name).
        expo = self.exposures
        expo_asof = (
            expo[expo["date"] <= date]
            .sort_values("date")
            .groupby("security_id")
            .tail(1)
            .set_index("security_id")
        )

        ids = [s for s in universe_ids if s in expo_asof.index]
        B = expo_asof.loc[ids, self.factor_names].values.astype(float)
        Fmat = F.loc[self.factor_names, self.factor_names].values
        d = np.array([d_series.get(s, np.nanmedian(d_series.values)) for s in ids], dtype=float)
        return RiskModelSnapshot(date, ids, self.factor_names, B, Fmat, d)
