"""Gradient-boosted-tree alpha model — the ML quants actually use.

Instead of the linear, IC-weighted blend, we let a gradient-boosted tree learn a
*nonlinear* map from the 6 cleaned signals to forward returns. Trees capture
interactions the linear model can't ("cheap AND high-momentum is special";
"low-vol only pays when quality is high"). This is the realistic ML for this
field — NOT deep learning (too little signal, too much noise for that).

THE critical detail: **walk-forward training, no look-ahead.** We never let the
model see the future. We retrain periodically on an *expanding window* of strictly
past data and predict only the block that follows:

    train on [start .. D)   ->   predict [D .. D+retrain_every)
    then slide D forward and retrain.

  * FEATURES (inputs)  = the 6 cleaned signal z-scores (what we know today).
  * TARGET  (output)   = next-period return, demeaned cross-sectionally each day
                         (who out/under-performs their peers — not market drift).

Output is one `alpha` per name per day — same shape as the linear model, so the
optimizer and backtest don't care which produced it.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from alphaforge.alpha.combine import clean_signal_panel
from alphaforge.config import Config
from alphaforge.logging import get_logger
from alphaforge.storage.warehouse import Warehouse

log = get_logger(__name__)


def _make_model():
    """A small, regularized gradient-boosted tree (sklearn — zero extra deps).

    Shallow trees + many small steps + strong L2 = robust on noisy financial data.
    HistGradientBoostingRegressor is fast and handles missing values natively.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.03,
        max_iter=300,
        l2_regularization=1.0,
        min_samples_leaf=200,
        early_stopping=False,
        random_state=0,
    )


def _prepare(cfg: Config, wh: Warehouse) -> tuple[pd.DataFrame, list[str]]:
    """Build the (features + target) frame: cleaned signals + demeaned fwd return."""
    clean, signal_cols = clean_signal_panel(cfg, wh)
    px = wh.read_table("prices", columns=["date", "security_id", "ret"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["security_id", "date"])
    px["fwd_ret"] = px.groupby("security_id")["ret"].shift(-1)
    df = clean.merge(px[["date", "security_id", "fwd_ret"]], on=["date", "security_id"])
    # TARGET: next-day return minus the cross-sectional mean that day.
    df["target"] = df["fwd_ret"] - df.groupby("date")["fwd_ret"].transform("mean")
    return df, signal_cols


def _walk_forward(
    df: pd.DataFrame, signal_cols: list[str], retrain_every: int, min_train: int
) -> Iterator[tuple[object, pd.DataFrame, pd.DataFrame]]:
    """Yield (fitted_model, train_block, test_block) for each walk-forward step.

    The model is trained ONLY on dates strictly before the test block — the
    look-ahead firewall. Shared by both prediction and feature-importance.
    """
    dates = np.array(sorted(df["date"].unique()))
    i = min_train
    while i < len(dates):
        block = dates[i : i + retrain_every]
        cutoff = block[0]
        train = df[df["date"] < cutoff].dropna(subset=[*signal_cols, "target"])
        test = df[df["date"].isin(block)].dropna(subset=signal_cols)
        i += retrain_every
        if len(train) < 1000 or test.empty:
            continue
        model = _make_model()
        model.fit(train[signal_cols].values, train["target"].values)
        yield model, train, test


def build_ml_alphas(cfg: Config, wh: Warehouse) -> pd.DataFrame:
    """Walk-forward gradient-boosted alpha. Returns date, security_id, <signals>, fvol, alpha."""
    df, signal_cols = _prepare(cfg, wh)
    preds: list[pd.DataFrame] = []
    n_models = 0
    for model, _train, test in _walk_forward(
        df, signal_cols, cfg.alpha.retrain_every, cfg.alpha.min_train
    ):
        out = test[["date", "security_id", *signal_cols, "fvol"]].copy()
        out["alpha"] = model.predict(test[signal_cols].values)
        preds.append(out)
        n_models += 1

    if not preds:
        raise RuntimeError("ML alpha produced no predictions — check min_train vs history length")
    result = pd.concat(preds, ignore_index=True).sort_values(["date", "security_id"])
    log.info("alpha.gbt.done", models_trained=n_models, rows=len(result))
    return result.reset_index(drop=True)


def ml_feature_importance(cfg: Config, wh: Warehouse, n_repeats: int = 4) -> pd.DataFrame:
    """Which signals did the model actually rely on? (out-of-sample permutation).

    For each walk-forward block we measure *permutation importance* on the
    held-out TEST set: shuffle one feature's values and see how much the model's
    accuracy drops. A big drop = the model leaned on that signal; ~0 = it was
    ignored. This is model-agnostic and uses only out-of-sample data, so it's an
    honest read on which factors carry the edge.
    """
    from sklearn.inspection import permutation_importance

    df, signal_cols = _prepare(cfg, wh)
    rows: list[np.ndarray] = []
    for model, _train, test in _walk_forward(
        df, signal_cols, cfg.alpha.retrain_every, cfg.alpha.min_train
    ):
        t = test.dropna(subset=[*signal_cols, "target"])
        if len(t) < 200:
            continue
        r = permutation_importance(
            model, t[signal_cols].values, t["target"].values,
            n_repeats=n_repeats, random_state=0, scoring="r2",
        )
        rows.append(r.importances_mean)

    if not rows:
        return pd.DataFrame(columns=["importance", "importance_pct"])
    mean_imp = np.vstack(rows).mean(axis=0)
    imp = pd.Series(mean_imp, index=signal_cols).clip(lower=0)  # negative = noise -> 0
    total = imp.sum() or 1.0
    out = pd.DataFrame({"importance": imp, "importance_pct": imp / total})
    return out.sort_values("importance", ascending=False)
