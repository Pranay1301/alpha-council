"""Signal strategies. Each returns a long/flat series (1 = enter long,
0 = no position) aligned to the input index, computed from data up to and
including that bar's close. Entries execute at the NEXT bar's open in the
backtester, so no signal can see the future.

Strategies are long-only: this tool suggests spot-style trades, and
short-selling/leverage is deliberately out of scope.
"""

from __future__ import annotations

import pandas as pd

from .indicators import atr, bollinger, ema, rsi


def trend_following(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                    rsi_cap: float = 70.0) -> pd.Series:
    """Long while EMA(fast) > EMA(slow) and RSI is not overbought."""
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)
    r = rsi(df["close"])
    sig = ((fast_ema > slow_ema) & (r < rsi_cap)).astype(float)
    return sig.fillna(0.0)


def mean_reversion(df: pd.DataFrame, rsi_window: int = 14,
                   oversold: float = 30.0, window: int = 20) -> pd.Series:
    """Long when RSI is oversold AND price is under the lower Bollinger band."""
    r = rsi(df["close"], rsi_window)
    _, _, lower = bollinger(df["close"], window)
    sig = ((r < oversold) & (df["close"] < lower)).astype(float)
    return sig.fillna(0.0)


def breakout(df: pd.DataFrame, lookback: int = 20,
             vol_mult: float = 1.5) -> pd.Series:
    """Long on a close above the prior `lookback` high with a volume surge."""
    prior_high = df["high"].shift(1).rolling(lookback).max()
    vol_ok = df["volume"] > df["volume"].rolling(lookback).mean() * vol_mult
    sig = ((df["close"] > prior_high) & vol_ok).astype(float)
    return sig.fillna(0.0)


STRATEGIES = {
    "trend_following": trend_following,
    "mean_reversion": mean_reversion,
    "breakout": breakout,
}
