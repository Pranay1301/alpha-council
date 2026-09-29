import numpy as np
import pandas as pd

from alphacouncil.optimize import walk_forward
from alphacouncil.strategies import STRATEGIES


def _trendy(n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(np.random.default_rng(7).normal(0.3, 1.5, n))
    return pd.DataFrame({"open": c - 0.2, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.full(n, 100.0)}, index=idx)


def test_multi_fold_aggregation_keys():
    df = _trendy()
    wf = walk_forward(df, "trend_following", STRATEGIES["trend_following"],
                      {"fast": [10], "slow": [40], "rsi_cap": [65.0]},
                      n_folds=3, min_trades=1)
    assert wf is not None
    tm = wf.test_metrics
    for k in ("win_rate", "profit_factor", "max_drawdown", "sharpe",
              "trades", "folds", "profitable_windows", "param_stability"):
        assert k in tm, k
    assert tm["folds"] == 3
    assert 0.0 <= tm["profitable_windows"] <= 1.0
    assert 0.0 <= tm["param_stability"] <= 1.0
