"""Shared fixtures: a tiny in-memory lake so tests run in <1s without the CLI."""
from __future__ import annotations

import pytest

from alphaforge.config import load_config
from alphaforge.data.loaders import ingest
from alphaforge.storage.warehouse import Warehouse


@pytest.fixture(scope="session")
def small_cfg():
    cfg = load_config("config/backtest.yaml")
    # Shrink everything so the suite is fast.
    cfg.universe.n_names = 60
    cfg.universe.top_by_adv = 40
    cfg.run.start_date = "2018-01-02"
    cfg.run.end_date = "2019-12-31"
    return cfg


@pytest.fixture(scope="session")
def lake(tmp_path_factory, small_cfg):
    root = tmp_path_factory.mktemp("lake")
    small_cfg.storage.lake_path = str(root)
    wh = Warehouse(str(root))
    ingest(small_cfg, wh)
    return wh
