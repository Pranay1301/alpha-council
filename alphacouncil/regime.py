"""Market-regime engine.

Every bar is classified into exactly one regime, so the desk can say WHERE
a strategy's edge historically appeared instead of quoting one blended
backtest number. Hierarchy (first match wins): capitulation, breakout,
high/low volatility, trending up/down, ranging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import _wilder_rma, atr, ema

TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
RANGING = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"
BREAKOUT = "BREAKOUT"
CAPITULATION = "CAPITULATION"


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    atrv = _wilder_rma(pd.concat([(high - low),
                                  (high - close.shift()).abs(),
                                  (low - close.shift()).abs()], axis=1).max(axis=1),
                       window)
    plus_di = 100 * _wilder_rma(plus_dm, window) / atrv
    minus_di = 100 * _wilder_rma(minus_dm, window) / atrv
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_rma(dx, window)


def classify(df: pd.DataFrame) -> pd.Series:
    """Per-bar regime labels, computed from data up to that bar only."""
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    adxv = adx(high, low, close)
    ema_fast, ema_slow = ema(close, 20), ema(close, 50)
    atr_pct = atr(high, low, close) / close
    atr_med = atr_pct.rolling(90, min_periods=30).median()
    prior_max = close.shift().rolling(20, min_periods=10).max()
    vol_avg = vol.rolling(20, min_periods=10).mean()
    drop = close.diff()

    regime = pd.Series(RANGING, index=df.index)
    regime[(adxv >= 25) & (ema_fast > ema_slow)] = TRENDING_UP
    regime[(adxv >= 25) & (ema_fast <= ema_slow)] = TRENDING_DOWN
    regime[atr_pct > 1.5 * atr_med] = HIGH_VOLATILITY
    regime[atr_pct < 0.6 * atr_med] = LOW_VOLATILITY
    regime[(close > prior_max) & (vol > 1.5 * vol_avg)] = BREAKOUT
    regime[(drop < -3 * atr(high, low, close)) & (vol > 2 * vol_avg)] = CAPITULATION
    regime[adxv.isna()] = RANGING
    return regime


def trades_by_regime(df: pd.DataFrame, trades) -> dict:
    """Bucket closed trades by the regime on their entry bar."""
    labels = classify(df)
    buckets: dict[str, list[float]] = {}
    for t in trades:
        if t.exit is None:
            continue
        loc = labels.index.get_loc(t.entry_time)
        reg = labels.iloc[min(loc, len(labels) - 1)]
        buckets.setdefault(reg, []).append(t.pnl_pct)
    out = {}
    for reg, pnls in buckets.items():
        p = np.array(pnls)
        wins = p[p > 0].sum()
        losses = -p[p <= 0].sum()
        out[reg] = {"trades": len(p),
                    "win_rate": float((p > 0).mean()),
                    "profit_factor": float(wins / losses) if losses > 0 else float("inf")}
    return out
