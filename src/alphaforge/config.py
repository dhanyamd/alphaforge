"""Typed configuration. Config is code — it is versioned, validated, diffable.

We parse `config/backtest.yaml` into pydantic models so that a typo like
`risk_avesion: 5` fails *loudly at load time* instead of silently trading the
wrong book. Every block of the machine pulls its parameters from here.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RunCfg(BaseModel):
    name: str = "baseline"
    seed: int = 7
    start_date: str
    end_date: str
    rebalance: str = "W-FRI"


class UniverseCfg(BaseModel):
    n_names: int = 200
    min_adv_usd: float = 1_000_000
    top_by_adv: int = 150


class DataCfg(BaseModel):
    source: str = "synthetic"
    vendors: list[str] = Field(default_factory=lambda: ["prices", "fundamentals", "altdata"])


class SignalCfg(BaseModel):
    name: str
    ic: float = 0.03           # prior information coefficient (signal "skill")
    enabled: bool = True


class AlphaCfg(BaseModel):
    # "linear" = classic IC-weighted Grinold blend; "gbt" = gradient-boosted trees
    # (walk-forward trained). Both produce the same panel shape.
    model: str = "linear"
    retrain_every: int = 63    # (gbt) retrain cadence in trading days (~quarterly)
    min_train: int = 252       # (gbt) minimum warmup history before first prediction


class RiskCfg(BaseModel):
    halflife_cov: int = 90
    halflife_var: int = 60
    shrinkage: float = 0.20
    lookback: int = 252


class PortfolioCfg(BaseModel):
    risk_aversion: float = 5.0
    cost_aversion: float = 1.0
    max_weight: float = 0.05
    max_gross: float = 1.0
    long_only: bool = True
    sector_neutral: bool = True
    max_turnover: float = 0.40


class ExecutionCfg(BaseModel):
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    impact_coef: float = 0.10
    participation: float = 0.10


class StorageCfg(BaseModel):
    lake_path: str = "data_lake"


class Config(BaseModel):
    run: RunCfg
    universe: UniverseCfg = UniverseCfg()
    data: DataCfg = DataCfg()
    alpha: AlphaCfg = AlphaCfg()
    signals: list[SignalCfg] = Field(default_factory=list)
    risk: RiskCfg = RiskCfg()
    portfolio: PortfolioCfg = PortfolioCfg()
    execution: ExecutionCfg = ExecutionCfg()
    storage: StorageCfg = StorageCfg()

    @property
    def enabled_signals(self) -> list[SignalCfg]:
        return [s for s in self.signals if s.enabled]


DEFAULT_CONFIG_PATH = Path("config/backtest.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate the YAML config."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config(**raw)
