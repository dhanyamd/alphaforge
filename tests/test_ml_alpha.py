"""The gradient-boosted alpha must be walk-forward (no look-ahead) and well-formed."""
from __future__ import annotations

import numpy as np

from alphaforge.alpha.ml_model import build_ml_alphas


def test_gbt_alpha_is_walk_forward_and_finite(lake, small_cfg):
    small_cfg.storage.lake_path = lake.root
    small_cfg.alpha.model = "gbt"
    small_cfg.alpha.min_train = 200          # small lake -> shorter warmup
    small_cfg.alpha.retrain_every = 40
    panel = build_ml_alphas(small_cfg, lake)

    assert not panel.empty
    for col in ["date", "security_id", "alpha"]:
        assert col in panel.columns
    assert np.isfinite(panel["alpha"]).all(), "alpha has NaN/inf"

    # Walk-forward: there can be NO predictions before the warmup window, because
    # the model is only allowed to predict AFTER training on strictly past data.
    dates = np.sort(panel["date"].unique())
    all_dates = np.sort(lake.read_table("prices", columns=["date"])["date"].unique())
    first_pred = np.where(all_dates == dates[0])[0][0]
    assert first_pred >= small_cfg.alpha.min_train - 1, "predicted before warmup — look-ahead!"
