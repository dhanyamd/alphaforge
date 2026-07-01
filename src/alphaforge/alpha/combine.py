"""Combine many refined alphas into one expected-return vector.

Multi-factor = run many alphas at once, each ideally an independent edge, then
blend them. We weight each factor by its configured IC (skill): a more skillful
signal gets more say. The blended alpha is the optimizer's expected-return input.

This module exposes:
  * `clean_signal_panel` — the shared front-end (raw signals -> hygiene -> +vol),
    reused by BOTH the linear blend and the ML (gradient-boosted) alpha model.
  * `build_alphas` — the classic linear, IC-weighted Grinold blend.
  * `build_alpha_panel` — a dispatcher that picks linear vs ML from config.
"""
from __future__ import annotations

import pandas as pd

from alphaforge.alpha.refine import clean_panel, forecast_vol, refine
from alphaforge.config import Config
from alphaforge.signals.library import compute_all_signals
from alphaforge.storage.warehouse import Warehouse


def clean_signal_panel(cfg: Config, wh: Warehouse) -> tuple[pd.DataFrame, list[str]]:
    """Shared front-end: raw signals -> cross-sectional hygiene -> attach fvol.

    Returns (panel, signal_cols) where panel has [date, security_id, <clean
    signals>, fvol]. These cleaned z-scores are the FEATURES both the linear and
    the ML alpha models consume.
    """
    signal_cols = [s.name for s in cfg.enabled_signals]
    raw = compute_all_signals(wh)
    static = wh.read_table("security_static")
    clean = clean_panel(raw, static, signal_cols)

    px = wh.read_table("prices", columns=["date", "security_id", "ret"])
    px["date"] = pd.to_datetime(px["date"])
    fvol = forecast_vol(px)
    clean = clean.merge(fvol, on=["date", "security_id"], how="left")
    return clean, signal_cols


def build_alphas(cfg: Config, wh: Warehouse) -> pd.DataFrame:
    """Linear alpha: cross-sectional hygiene -> Grinold refine -> IC-weighted blend."""
    clean, signal_cols = clean_signal_panel(cfg, wh)
    ic_map = {s.name: s.ic for s in cfg.enabled_signals}

    alpha = pd.Series(0.0, index=clean.index)
    weight_sum = sum(abs(ic) for ic in ic_map.values()) or 1.0
    for col in signal_cols:
        ic = ic_map[col]
        contrib = refine(clean[col], clean["fvol"], ic)
        alpha = alpha + (abs(ic) / weight_sum) * contrib
    clean["alpha"] = alpha

    keep = ["date", "security_id", *signal_cols, "fvol", "alpha"]
    return clean[keep].sort_values(["date", "security_id"]).reset_index(drop=True)


def build_alpha_panel(cfg: Config, wh: Warehouse) -> pd.DataFrame:
    """Dispatcher: build the alpha panel using the model named in config.

    `alpha.model: "linear"` (default) or `"gbt"` (gradient-boosted trees). Both
    return the same shape so the backtest/optimizer don't care which was used.
    """
    if cfg.alpha.model == "gbt":
        from alphaforge.alpha.ml_model import build_ml_alphas
        return build_ml_alphas(cfg, wh)
    return build_alphas(cfg, wh)
