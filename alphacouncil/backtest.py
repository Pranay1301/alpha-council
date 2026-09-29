"""Event-driven long-only backtester with stop-loss / take-profit exits.

Execution model (deliberately conservative):
- Signals are computed at bar close; entries fill at the NEXT bar's open
  plus slippage. No lookahead.
- Stop/target checks begin on the bar AFTER the entry bar: the entry bar's
  range includes prices from before the entry fill, so it can never trigger
  an exit.
- While in a trade, each bar is checked against the stop first: if a bar's
  range touches BOTH stop and target, the stop is assumed hit first. This
  under-reports performance rather than flattering it.
- Fees are charged on both entry and exit (default 10 bps/side, like a
  major exchange's spot taker fee), slippage on entry and exit.
- Exits: stop-loss, take-profit, signal-off, or max holding bars.

The equity curve is a genuine bar-by-bar portfolio curve: it is flat while
no position is held and marks the open position to market on every bar, so
Sharpe ratio and drawdown measure the strategy's actual path, not a step
function stitched from closed-trade P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr

PERIODS_PER_YEAR = {"1d": 365, "1h": 365 * 24}


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    entry: float
    exit: float | None
    stop: float
    target: float
    exit_reason: str = ""
    pnl_pct: float = 0.0        # after fees, as a fraction (0.02 = +2%)
    r_multiple: float = 0.0     # pnl in units of initial risk
    mae: float = 0.0            # max adverse excursion while in trade
    mfe: float = 0.0            # max favorable excursion while in trade


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series | None = None   # bar-by-bar portfolio equity, 1.0 start
    interval: str = "1d"

    def metrics(self) -> dict:
        closed = [t for t in self.trades if t.exit is not None]
        if not closed:
            return {"trades": 0}
        pnls = np.array([t.pnl_pct for t in closed])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gross_win = wins.sum()
        gross_loss = -losses.sum()
        eq = self.equity.dropna() if self.equity is not None else None
        max_dd = float(((eq.cummax() - eq) / eq.cummax()).max()) if eq is not None and len(eq) else 0.0
        ppy = PERIODS_PER_YEAR.get(self.interval, 365)
        sharpe = 0.0
        if eq is not None and len(eq) > 2:
            rets = eq.pct_change().dropna()
            if rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * np.sqrt(ppy))
        return {
            "trades": len(closed),
            "win_rate": float((pnls > 0).mean()),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1) if eq is not None and len(eq) else float(pnls.sum()),
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "avg_r": float(np.mean([t.r_multiple for t in closed])),
        }


def run(df: pd.DataFrame, signals: pd.Series, *, stop_atr: float = 2.0,
        rr: float = 2.0, fee_bps: float = 10.0, slippage_bps: float = 5.0,
        max_hold: int = 30, exit_on_signal_off: bool = True,
        interval: str = "1d") -> BacktestResult:
    """Backtest a long/flat signal series on an OHLCV frame.

    stop_atr: stop distance = stop_atr * ATR(14) at entry.
    rr: take-profit distance = rr * stop distance (risk multiple).
    """
    signals = signals.reindex(df.index).fillna(0.0)
    atrv = atr(df["high"], df["low"], df["close"]).to_numpy()
    sig = signals.to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    n = len(df)
    fee = fee_bps / 1e4
    slip = slippage_bps / 1e4

    res = BacktestResult(interval=interval)
    eq = np.ones(n)
    cur = 1.0                 # current portfolio equity
    entry_eq = 0.0            # equity at entry, net of entry fee
    pos: Trade | None = None
    entry_bar = -1
    bars_held = 0

    for i in range(n):
        if pos is None:
            eq[i] = cur
            if i < n - 1 and sig[i] >= 1.0 and not np.isnan(atrv[i]):
                risk = stop_atr * atrv[i]
                if risk <= 0:
                    continue
                entry_price = o[i + 1] * (1 + slip)
                pos = Trade(entry_time=idx[i + 1], exit_time=None,
                            entry=entry_price, exit=None,
                            stop=entry_price - risk,
                            target=entry_price + rr * risk)
                entry_bar = i + 1
                bars_held = 0
                entry_eq = cur * (1 - fee)   # entry fee hits equity now
        else:
            bars_held += 1
            exit_price, reason = None, ""
            if i > entry_bar:
                # stop/target/signal checks only on bars after the entry bar
                if lo[i] <= pos.stop:
                    exit_price, reason = pos.stop * (1 - slip), "stop_loss"
                elif h[i] >= pos.target:
                    exit_price, reason = pos.target * (1 - slip), "take_profit"
                elif exit_on_signal_off and sig[i] < 1.0:
                    exit_price, reason = c[i] * (1 - slip), "signal_off"
                elif bars_held >= max_hold:
                    exit_price, reason = c[i] * (1 - slip), "max_hold"
            # excursions tracked bar-by-bar while the trade is open
            pos.mae = min(pos.mae, lo[i] / pos.entry - 1)
            pos.mfe = max(pos.mfe, h[i] / pos.entry - 1)
            if exit_price is not None:
                pos.exit = exit_price
                pos.exit_time = idx[i]
                pos.exit_reason = reason
                pos.pnl_pct = (pos.exit / pos.entry) * (1 - fee) / (1 + fee) - 1
                pos.r_multiple = pos.pnl_pct / ((pos.entry - pos.stop) / pos.entry)
                res.trades.append(pos)
                cur = entry_eq * (pos.exit / pos.entry) * (1 - fee)
                pos = None
                eq[i] = cur
            else:
                eq[i] = entry_eq * (c[i] / pos.entry)  # mark to market
    if pos is not None:  # close at the final close, marked as open-end
        pos.exit = c[-1] * (1 - slip)
        pos.exit_time = idx[-1]
        pos.exit_reason = "end_of_data"
        pos.pnl_pct = (pos.exit / pos.entry) * (1 - fee) / (1 + fee) - 1
        pos.r_multiple = pos.pnl_pct / ((pos.entry - pos.stop) / pos.entry)
        res.trades.append(pos)
        eq[-1] = entry_eq * (pos.exit / pos.entry) * (1 - fee)

    res.equity = pd.Series(eq, index=idx)
    return res
