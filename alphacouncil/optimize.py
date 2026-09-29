"""Walk-forward parameter selection.

The honest way to "train" a strategy on historical data: choose parameters
on a training window only, then report metrics on a held-out test window
the optimizer never saw. A strategy that only works in-sample is
overfitting, and the report should say so - that is why test metrics, not
train metrics, appear on every trade suggestion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run


@dataclass
class WalkForwardResult:
    strategy: str
    best_params: dict
    train: BacktestResult
    test: BacktestResult
    train_metrics: dict
    test_metrics: dict


def walk_forward(df: pd.DataFrame, strategy_name: str, strategy_fn,
                 param_grid: dict, *, train_frac: float = 0.7,
                 min_trades: int = 5, interval: str = "1d",
                 bt_kwargs: dict | None = None) -> WalkForwardResult | None:
    """Grid-search params on the train split, evaluate once on the test split.

    Returns None when no parameter set produces enough train trades to be
    meaningful - the honest answer is sometimes "no edge here".
    """
    bt_kwargs = bt_kwargs or {}
    split = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    keys = list(param_grid)
    best, best_score, best_train = None, -1.0, None
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        signals = strategy_fn(train_df, **params)
        res = run(train_df, signals, interval=interval, **bt_kwargs)
        m = res.metrics()
        if m.get("trades", 0) < min_trades:
            continue
        pf = min(m["profit_factor"], 5.0)  # cap so one lucky run doesn't dominate
        score = pf * m["win_rate"]
        if score > best_score:
            best, best_score, best_train = params, score, res
    if best is None:
        return None
    test_signals = strategy_fn(test_df, **best)
    test_res = run(test_df, test_signals, interval=interval, **bt_kwargs)
    return WalkForwardResult(strategy_name, best, best_train, test_res,
                             best_train.metrics(), test_res.metrics())
