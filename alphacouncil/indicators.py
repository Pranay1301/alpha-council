"""Technical indicators, implemented from scratch on numpy/pandas.

Conventions:
- Wilder smoothing (alpha = 1/n) for RSI and ATR, matching TradingView.
- EMA with adjust=False (standard charting behavior).
- All functions take/return pandas Series indexed like the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average: SMA seed over the first n valid values,
    then the recursive rma[i] = (rma[i-1]*(n-1) + s[i]) / n. This is the
    smoothing TradingView uses for RSI and ATR."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    start = 0
    while start < len(vals) and np.isnan(vals[start]):
        start += 1
    if len(vals) - start < n:
        return pd.Series(out, index=s.index)
    acc = vals[start:start + n].mean()
    out[start + n - 1] = acc
    for i in range(start + n, len(vals)):
        acc = (acc * (n - 1) + vals[i]) / n
        out[i] = acc
    return pd.Series(out, index=s.index)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. All-gains window -> 100, all-losses -> 0."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_rma(gain, window)
    avg_loss = _wilder_rma(loss, window)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Returns (middle, upper, lower)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return mid, mid + num_std * std, mid - num_std * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Average True Range - used for stop-loss sizing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder_rma(tr, window)
