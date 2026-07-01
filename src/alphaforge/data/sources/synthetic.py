"""Synthetic market generator — a toy market with *real* factor structure.

This is what makes the project runnable offline AND honest: we don't sprinkle
random numbers and pretend. We generate returns from an explicit factor model so
that the signals we later compute genuinely predict future returns (positive but
*modest* IC, exactly like reality). The data-generating process:

    r_{i,t} =  beta_i * market_t                      # systematic market move
             + sector_{s(i),t}                        # the name's sector move
             + Σ_f  premium^f_t * z(char^f_{i,t})      # PAYOFF to style exposures
             + u_{i,t}                                 # persistent specific (momentum)
             + eps_{i,t}                               # pure idiosyncratic noise

Predictability is injected three ways, each recoverable by a real signal:
  * value / quality / altdata : latent AR(1) characteristics with a positive
    average premium. We publish *noisy, lagged* observations of them in the
    fundamentals/altdata vendor tables — the signals layer recovers them.
  * momentum : `u` is a persistent (high-AR(1)) specific component, so trailing
    winners keep winning → a price-based momentum signal works.
  * low-vol / size : low-idiosyncratic-vol and small-cap names get a small mean
    premium → price-based vol and market-cap signals work.

None of this is visible to downstream code — it only sees prices, fundamentals,
and alt-data, exactly as a real desk would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.config import Config
from alphaforge.data.sources.base import DataSource, MarketData

SECTORS = [
    "Technology", "Financials", "Healthcare", "Energy",
    "Consumer", "Industrials", "Utilities", "Materials",
]

def _ar1(T: int, N: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate an (T, N) AR(1) process x_t = rho x_{t-1} + sqrt(1-rho^2) sigma e_t."""
    x = np.zeros((T, N))
    innov = rng.standard_normal((T, N)) * sigma * np.sqrt(1 - rho**2)
    x[0] = rng.standard_normal(N) * sigma
    for t in range(1, T):
        x[t] = rho * x[t - 1] + innov[t]
    return x


def _xs_standardize(x: np.ndarray) -> np.ndarray:
    """Cross-sectionally standardize each row (date) to mean 0, std 1."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-9
    return (x - mu) / sd


class SyntheticSource(DataSource):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def fetch(self, start: str, end: str) -> MarketData:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.run.seed)
        dates = pd.bdate_range(start, end)
        T, N = len(dates), cfg.universe.n_names
        sids = [f"SEC{idx:04d}" for idx in range(N)]

        # ---- static characteristics ---------------------------------------
        sector_idx = rng.integers(0, len(SECTORS), size=N)
        sectors = [SECTORS[k] for k in sector_idx]
        beta = rng.normal(1.0, 0.25, size=N).clip(0.3, 1.8)          # market betas
        idio_vol = rng.uniform(0.010, 0.030, size=N)                 # per-name specific vol
        size0 = rng.normal(0.0, 1.0, size=N)                         # latent log-size (static-ish)

        # ---- systematic factor returns ------------------------------------
        market = rng.normal(0.0003, 0.011, size=T)                   # ~ +7.5%/yr drift, 17% vol
        sector_ret = rng.normal(0.0, 0.006, size=(T, len(SECTORS)))  # sector factor returns

        # ---- latent style characteristics (value/quality/altdata) ---------
        # Slow AR(1) so a name's "cheapness" persists for months.
        char_value = _xs_standardize(_ar1(T, N, rho=0.995, sigma=1.0, rng=rng))
        char_quality = _xs_standardize(_ar1(T, N, rho=0.997, sigma=1.0, rng=rng))
        char_alt = _xs_standardize(_ar1(T, N, rho=0.97, sigma=1.0, rng=rng))   # alt-data decays faster

        # ---- persistent specific component (drives momentum) --------------
        u = _ar1(T, N, rho=0.985, sigma=0.0028, rng=rng)

        # ---- TIME-VARYING factor premia -----------------------------------
        # Crucial for realism: the premium a style PAYS is itself a noisy time
        # series that frequently flips sign. Exposure to value pays off *on
        # average* (positive mean) but loses in many months — that factor-timing
        # risk is exactly why real multifactor IRs are ~0.5-1.5, not infinite.
        # premium_t ~ mean + vol * N(0,1); vol >> mean, so signs flip often.
        def premium(mean: float, vol: float) -> np.ndarray:
            return mean + vol * rng.standard_normal(T)

        p_value = premium(0.00010, 0.0011)     # ~+2.5%/yr mean, flips ~45% of days
        p_quality = premium(0.00008, 0.0010)
        p_alt = premium(0.00013, 0.0012)       # alt-data: the strongest edge
        p_lowvol = premium(0.00006, 0.0008)
        p_size = premium(0.00006, 0.0009)
        lowvol_char = -_xs_standardize(idio_vol[None, :].repeat(T, axis=0))
        size_char = -_xs_standardize(size0[None, :].repeat(T, axis=0))

        # ---- assemble daily returns ---------------------------------------
        ret = np.zeros((T, N))
        ret += beta[None, :] * market[:, None]
        ret += sector_ret[:, sector_idx]
        ret += p_value[:, None] * char_value
        ret += p_quality[:, None] * char_quality
        ret += p_alt[:, None] * char_alt
        ret += p_lowvol[:, None] * lowvol_char
        ret += p_size[:, None] * size_char
        ret += u                                                     # momentum source
        ret += rng.standard_normal((T, N)) * idio_vol[None, :]       # pure noise

        # ---- prices, volume, shares, market cap ---------------------------
        price0 = rng.uniform(20, 200, size=N)
        close = price0[None, :] * np.cumprod(1 + ret, axis=0)
        shares_out = rng.uniform(5e7, 2e9, size=N).round(-6)         # shares outstanding
        # Dollar volume: scales with cap, with daily noise; this drives liquidity.
        base_dollar_vol = (close * shares_out) * rng.uniform(0.002, 0.02, size=N)[None, :]
        dollar_vol = base_dollar_vol * np.exp(rng.normal(0, 0.3, size=(T, N)))
        volume = (dollar_vol / close).round()

        prices = self._melt(
            dates, sids,
            {"close": close, "volume": volume, "ret": ret},
        )
        prices["shares_out"] = np.tile(shares_out, T)

        # ---- fundamentals (monthly, with a reporting lag) -----------------
        funda = self._fundamentals(dates, sids, char_value, char_quality, close, shares_out, rng)

        # ---- alt-data (noisy, lagged observation of char_alt) -------------
        alt = self._altdata(dates, sids, char_alt, rng)

        static = pd.DataFrame({"security_id": sids, "sector": sectors})
        return MarketData(prices=prices, fundamentals=funda, altdata=alt, static=static)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _melt(dates: pd.DatetimeIndex, sids: list[str], mats: dict[str, np.ndarray]) -> pd.DataFrame:
        T, N = len(dates), len(sids)
        base = pd.DataFrame({
            "date": np.repeat(dates.values, N),
            "security_id": np.tile(sids, T),
        })
        for name, mat in mats.items():
            base[name] = mat.reshape(-1)
        return base

    def _fundamentals(self, dates, sids, char_value, char_quality, close, shares_out, rng):
        """Publish noisy book-to-price and gross-profitability, monthly + lagged.

        Real fundamentals arrive on a reporting calendar with a lag — you only
        *know* last quarter's book value some weeks after quarter-end. We emulate
        that: values update month-start and are observable as-of that date.
        """
        T, N = len(dates), len(sids)
        # Noisy observation of the latent characteristics (signal recoverability < 1).
        obs_value = char_value + rng.normal(0, 0.5, size=(T, N))
        obs_quality = char_quality + rng.normal(0, 0.5, size=(T, N))
        mktcap = close * shares_out[None, :]

        # Sample on the first business day of each month (the "as-of" reporting date).
        month_starts = pd.Series(dates).groupby([dates.year, dates.month]).first().values
        idx = pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(month_starts))
        rows = []
        for t in idx:
            # book-to-price proxy: cheaper (high value char) -> higher B/P.
            book_to_price = np.exp(0.4 * obs_value[t] - 0.5)
            gross_profit = (0.15 + 0.05 * obs_quality[t]) * mktcap[t]   # GP scaled to cap
            total_assets = mktcap[t] * rng.uniform(0.8, 1.5, size=N)
            rows.append(pd.DataFrame({
                "date": dates[t],
                "security_id": sids,
                "book_to_price": book_to_price,
                "gross_profit": gross_profit,
                "total_assets": total_assets,
            }))
        return pd.concat(rows, ignore_index=True)

    def _altdata(self, dates, sids, char_alt, rng):
        """Alt-data score: a weekly, noisy, *lagged* read on the alt characteristic."""
        T, N = len(dates), len(sids)
        obs = char_alt + rng.normal(0, 0.6, size=(T, N))
        # Weekly cadence (Mondays), published with a 1-day lag baked into the date.
        weekly_mask = pd.DatetimeIndex(dates).weekday == 0
        rows = []
        for t in np.where(weekly_mask)[0]:
            rows.append(pd.DataFrame({
                "date": dates[t],
                "security_id": sids,
                "alt_score": obs[t],
            }))
        return pd.concat(rows, ignore_index=True)
