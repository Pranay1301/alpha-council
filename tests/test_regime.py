import numpy as np
import pandas as pd

from alphacouncil.backtest import run
from alphacouncil.regime import CAPITULATION, RANGING, TRENDING_UP, classify, trades_by_regime


def _mk(closes, vol=100.0):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": np.full(len(c), vol)}, index=idx)


def test_trending_up_detected():
    df = _mk(list(np.linspace(100, 200, 120)))
    assert classify(df).iloc[-1] == TRENDING_UP


def test_flat_market_is_ranging():
    df = _mk([100 + (i % 2) * 0.1 for i in range(120)])
    assert classify(df).iloc[-1] == RANGING


def test_capitulation_detected():
    closes = list(np.linspace(100, 101, 100)) + [85.0]
    df = _mk(closes)
    df.loc[df.index[-1], "volume"] = 1000.0
    df.loc[df.index[-1], "low"] = 84.0
    df.loc[df.index[-1], "high"] = 101.0
    assert classify(df).iloc[-1] == CAPITULATION


def test_trades_bucketed_by_regime():
    df = _mk(list(np.linspace(100, 160, 120)))
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[30] = 1.0
    res = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=0, slippage_bps=0,
              exit_on_signal_off=False)
    buckets = trades_by_regime(df, res.trades)
    assert buckets
    assert sum(b["trades"] for b in buckets.values()) == 1
