"""Backtester correctness tests. Run: python tests/test_backtest.py (or pytest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphacouncil.backtest import run


def make_df(closes, spread=1.0):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": np.roll(closes, 1),
        "high": closes + spread,
        "low": closes - spread,
        "close": closes,
        "volume": np.full(len(closes), 1000.0),
    }, index=idx).assign(open=lambda d: d["open"].where(d.index > d.index[0], closes[0]))


def flat_then_up(n_flat=25, n_up=40, start=100.0, step=1.0):
    return [start] * n_flat + [start + i * step for i in range(1, n_up + 1)]


def test_entry_is_next_bar_open_no_lookahead():
    df = make_df(flat_then_up())
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[20] = 1.0  # signal at close of bar 20 -> entry at open of bar 21
    res = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=0, slippage_bps=0,
              exit_on_signal_off=False)
    t = res.trades[0]
    assert t.entry_time == df.index[21]
    assert t.entry == df["open"].iloc[21]


def test_take_profit_hit_in_rising_market():
    df = make_df(flat_then_up(step=1.0), spread=0.5)
    sig = pd.Series(1.0, index=df.index)
    res = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=0, slippage_bps=0,
              exit_on_signal_off=False)
    t = res.trades[0]
    assert t.exit_reason == "take_profit"
    assert abs(t.exit - t.target) < 1e-9
    assert t.pnl_pct > 0
    assert t.r_multiple > 1.5  # roughly rr=2


def test_stop_loss_hit_in_falling_market():
    df = make_df([100.0] * 25 + [100 - i for i in range(1, 40)], spread=0.5)
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[24] = 1.0
    res = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=0, slippage_bps=0,
              exit_on_signal_off=False)
    t = res.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.pnl_pct < 0
    assert t.r_multiple >= -1.3  # roughly -1R after slippage-free math


def test_stop_wins_when_bar_touches_both():
    # one violent bar spans both stop and target -> stop must be assumed
    closes = [100.0] * 25 + [101.0, 97.0, 96.0]
    df = make_df(closes, spread=0.5)
    df.loc[df.index[26], "high"] = 120.0
    df.loc[df.index[26], "low"] = 80.0
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[24] = 1.0
    res = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=0, slippage_bps=0,
              exit_on_signal_off=False)
    assert res.trades[0].exit_reason == "stop_loss"


def test_fees_reduce_pnl():
    df = make_df(flat_then_up(), spread=0.5)
    sig = pd.Series(1.0, index=df.index)
    free = run(df, sig, fee_bps=0, slippage_bps=0, exit_on_signal_off=False).trades[0]
    taxed = run(df, sig, fee_bps=10, slippage_bps=0, exit_on_signal_off=False).trades[0]
    assert taxed.pnl_pct < free.pnl_pct


def test_metrics_on_known_outcomes():
    df = make_df(flat_then_up() + [140 - i for i in range(30)] + flat_then_up(10, 20, 111, 1.0), spread=0.5)
    sig = pd.Series(1.0, index=df.index)
    res = run(df, sig, fee_bps=0, slippage_bps=0, exit_on_signal_off=False)
    m = res.metrics()
    assert m["trades"] == len(res.trades) >= 2
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["max_drawdown"] >= 0.0


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"pass: {name}")
    print("all backtest tests passed")
