"""Rolling multi-window walk-forward parameter selection.

The honest way to "train" a strategy on historical data: choose parameters
on a training window only, then report metrics on a held-out test window
the optimizer never saw. A single 70/30 split is still vulnerable to one
particular historical split, so this runs several rolling windows:

    fold 1: train [0 ..... 50%]  test [50% .. 66%]
    fold 2: train [0 ..... 66%]  test [66% .. 83%]
    fold 3: train [0 ..... 83%]  test [83% .. 100%]

and aggregates across them: median profit factor / Sharpe / win rate,
worst drawdown, total trades, share of profitable windows, and parameter
stability (how often the same parameters won). Test metrics - never train
metrics - appear on every trade suggestion.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run


@dataclass
class WalkForwardResult:
    strategy: str
    best_params: dict           # modal parameters across folds
    train: BacktestResult
    test: BacktestResult
    train_metrics: dict
    test_metrics: dict          # aggregated across folds (see module docstring)
    fold_params: list[dict] = field(default_factory=list)


def _score(m: dict) -> float:
    """Robustness-weighted selection score: rewards profit factor and
    Sharpe, discounts high drawdown and lucky low-trade runs."""
    pf = min(m.get("profit_factor", 0), 5.0)
    sharpe = max(m.get("sharpe", 0.0), 0.0)
    dd = min(m.get("max_drawdown", 1.0), 0.9)
    trades = m.get("trades", 0)
    consistency = min(trades / 20.0, 1.0)
    return (0.6 * pf + 0.4 * sharpe) * (1 - dd) * (0.5 + 0.5 * consistency)


def _optimize_fold(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   strategy_fn, param_grid: dict, *, min_trades: int,
                   interval: str, bt_kwargs: dict):
    keys = list(param_grid)
    best, best_score, best_train = None, -1.0, None
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        signals = strategy_fn(train_df, **params)
        res = run(train_df, signals, interval=interval, **bt_kwargs)
        m = res.metrics()
        if m.get("trades", 0) < min_trades:
            continue
        score = _score(m)
        if score > best_score:
            best, best_score, best_train = params, score, res
    if best is None:
        return None
    test_signals = strategy_fn(test_df, **best)
    test_res = run(test_df, test_signals, interval=interval, **bt_kwargs)
    return best, best_train, test_res


def walk_forward(df: pd.DataFrame, strategy_name: str, strategy_fn,
                 param_grid: dict, *, n_folds: int = 3,
                 train_frac: float = 0.5, min_trades: int = 5,
                 interval: str = "1d",
                 bt_kwargs: dict | None = None) -> WalkForwardResult | None:
    """Grid-search params per rolling fold, evaluate each on its held-out
    window, aggregate. Returns None when no fold yields a meaningful edge -
    the honest answer is sometimes "no edge here"."""
    bt_kwargs = bt_kwargs or {}
    n = len(df)
    test_len = int(n * (1 - train_frac) / n_folds)
    folds = []
    for k in range(n_folds):
        split = int(n * train_frac) + k * test_len
        if split + test_len > n:
            break
        fold = _optimize_fold(df.iloc[:split], df.iloc[split:split + test_len],
                              strategy_fn, param_grid, min_trades=min_trades,
                              interval=interval, bt_kwargs=bt_kwargs)
        if fold is not None:
            folds.append(fold)
    if not folds:
        return None

    params_list, train_res_list, test_res_list = zip(*folds)
    tms = [r.metrics() for r in test_res_list]
    trms = [r.metrics() for r in train_res_list]
    pfs = [m.get("profit_factor", 0) for m in tms]
    modal_params, modal_count = Counter(
        tuple(sorted(p.items())) for p in params_list).most_common(1)[0]

    test_metrics = {
        "win_rate": float(np.median([m.get("win_rate", 0) for m in tms])),
        "profit_factor": float(np.median(pfs)),
        "max_drawdown": float(max(m.get("max_drawdown", 0) for m in tms)),
        "sharpe": float(np.median([m.get("sharpe", 0) for m in tms])),
        "total_return": float(np.median([m.get("total_return", 0) for m in tms])),
        "avg_r": float(np.median([m.get("avg_r", 0) for m in tms])),
        "trades": int(sum(m.get("trades", 0) for m in tms)),
        "folds": len(folds),
        "profitable_windows": float(np.mean([pf > 1.0 for pf in pfs])),
        "param_stability": modal_count / len(folds),
    }
    train_metrics = {
        "win_rate": float(np.median([m.get("win_rate", 0) for m in trms])),
        "profit_factor": float(np.median([m.get("profit_factor", 0) for m in trms])),
        "max_drawdown": float(max(m.get("max_drawdown", 0) for m in trms)),
        "trades": int(sum(m.get("trades", 0) for m in trms)),
    }
    return WalkForwardResult(strategy_name, dict(modal_params),
                             train_res_list[-1], test_res_list[-1],
                             train_metrics, test_metrics,
                             fold_params=[dict(p) for p in params_list])
