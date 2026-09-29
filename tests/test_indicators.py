"""Indicator correctness tests. Run: python tests/test_indicators.py (or pytest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphacouncil.indicators import atr, bollinger, ema, macd, rsi, sma


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 60.0))           # strictly rising
    down = pd.Series(np.arange(60.0, 1.0, -1.0))   # strictly falling
    flat = pd.Series(np.full(60, 10.0))
    assert rsi(up).iloc[-1] == 100.0
    assert rsi(down).iloc[-1] == 0.0
    assert rsi(flat).iloc[-1] == 50.0
    s = rsi(pd.Series(np.random.default_rng(1).normal(100, 5, 500).cumsum() + 1000))
    assert ((s.dropna() >= 0) & (s.dropna() <= 100)).all()


def test_rsi_matches_hand_computed():
    # Wilder's RSI on a small known series
    closes = pd.Series([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 12, 13, 14, 13, 12, 13], dtype=float)
    r = rsi(closes, window=14)
    delta = closes.diff().iloc[1:15]
    gains = delta.clip(lower=0).sum() / 14
    losses = -delta.clip(upper=0).sum() / 14
    expected = 100 - 100 / (1 + gains / losses)
    assert abs(r.iloc[14] - expected) < 1e-9


def test_ema_converges_to_constant():
    s = pd.Series([5.0] * 100)
    assert abs(ema(s, 10).iloc[-1] - 5.0) < 1e-12


def test_macd_histogram_identity():
    close = pd.Series(np.random.default_rng(2).normal(0, 1, 300).cumsum() + 100)
    line, sig, hist = macd(close)
    assert np.allclose((line - sig).dropna(), hist.dropna())


def test_bollinger_width_positive_and_ordered():
    close = pd.Series(np.random.default_rng(3).normal(100, 3, 200))
    mid, up, lo = bollinger(close)
    assert ((up.dropna() >= mid.dropna()) & (mid.dropna() >= lo.dropna())).all()


def test_atr_nonnegative_and_bounds_tr():
    n = 200
    rng = np.random.default_rng(4)
    close = pd.Series(rng.normal(100, 2, n).cumsum() + 500)
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    a = atr(high, low, close)
    assert (a.dropna() >= 0).all()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    assert a.dropna().iloc[-1] <= tr.max()  # ATR is a weighted mean of TR


def test_sma_simple():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert sma(s, 2).iloc[-1] == 3.5


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"pass: {name}")
    print("all indicator tests passed")
