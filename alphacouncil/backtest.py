"""Event-driven long-only execution engine with stop-loss / take-profit exits.

There is ONE canonical trade simulation here - simulate_trade() - and every
consumer uses it: the backtester, the historical replay (trackrecord.py),
the persistent paper ledger (paper.py) and the council-effectiveness study
(effectiveness.py). Backtest and track record can never silently diverge.

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
    bars_held: int = 0          # bars from entry bar through exit bar


@dataclass
class SimResult:
    trade: Trade
    entry_bar: int              # array position of the entry bar
    exit_bar: int               # array position of the exit bar


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


def monitor_trade(entry: float, stop: float, target: float, entry_bar: int,
                  h, lo, c, *, fee_bps: float = 10.0, slippage_bps: float = 5.0,
                  max_hold: int = 30, sig=None, exit_on_signal_off: bool = True,
                  index=None, entry_time=None) -> SimResult:
    """Canonical trade monitoring: fixed entry/stop/target, walk bars forward.

    Stop/target/signal checks begin on the bar AFTER entry_bar (the entry
    bar's range predates the fill). A bar touching both stop and target
    counts as a stop. MAE/MFE bound the move through the exit bar; intrabar
    order is unknown, so those are bounds, not tick-level claims.

    Used directly by the paper ledger (stored levels) and by simulate_trade
    (levels derived from a signal bar). Never reimplement this loop elsewhere.
    """
    n = len(c)
    fee = fee_bps / 1e4
    slip = slippage_bps / 1e4
    trade = Trade(entry_time=entry_time if entry_time is not None else (index[entry_bar] if index is not None else entry_bar),
                  exit_time=None, entry=entry, exit=None,
                  stop=stop, target=target)
    for i in range(entry_bar, n):
        bars_held = i - entry_bar + 1
        exit_price, reason = None, ""
        if i > entry_bar:
            if lo[i] <= stop:
                exit_price, reason = stop * (1 - slip), "stop_loss"
            elif h[i] >= target:
                exit_price, reason = target * (1 - slip), "take_profit"
            elif exit_on_signal_off and sig is not None and sig[i] < 1.0:
                exit_price, reason = c[i] * (1 - slip), "signal_off"
            elif bars_held >= max_hold:
                exit_price, reason = c[i] * (1 - slip), "max_hold"
        trade.mae = min(trade.mae, lo[i] / entry - 1)
        trade.mfe = max(trade.mfe, h[i] / entry - 1)
        if exit_price is not None:
            trade.exit = exit_price
            trade.exit_time = index[i] if index is not None else i
            trade.exit_reason = reason
            trade.bars_held = bars_held
            break
    if trade.exit is None:  # still open at the end of the data: mark to last close
        i = n - 1
        trade.exit = c[i] * (1 - slip)
        trade.exit_time = index[i] if index is not None else i
        trade.exit_reason = "end_of_data"
        trade.bars_held = i - entry_bar + 1
    trade.pnl_pct = (trade.exit / trade.entry) * (1 - fee) / (1 + fee) - 1
    risk_frac = (trade.entry - trade.stop) / trade.entry
    trade.r_multiple = trade.pnl_pct / risk_frac if risk_frac > 0 else 0.0
    return SimResult(trade=trade, entry_bar=entry_bar, exit_bar=i)


def simulate_trade(o, h, lo, c, signal_bar: int, atr_value: float, *,
                   stop_atr: float = 2.0, rr: float = 2.0, fee_bps: float = 10.0,
                   slippage_bps: float = 5.0, max_hold: int = 30, sig=None,
                   exit_on_signal_off: bool = True, index=None) -> SimResult | None:
    """THE canonical trade: signal at `signal_bar`'s close, entry at the NEXT
    bar's open + slippage, stop = entry - stop_atr*ATR(signal bar),
    target = entry + rr*risk, then monitor_trade(). Returns None when the
    signal bar has no next bar to enter on.
    """
    n = len(c)
    entry_bar = signal_bar + 1
    if entry_bar >= n:
        return None
    risk = stop_atr * atr_value
    if risk <= 0 or np.isnan(risk):
        return None
    slip = slippage_bps / 1e4
    entry = o[entry_bar] * (1 + slip)
    return monitor_trade(entry, entry - risk, entry + rr * risk, entry_bar,
                         h, lo, c, fee_bps=fee_bps, slippage_bps=slippage_bps,
                         max_hold=max_hold, sig=sig,
                         exit_on_signal_off=exit_on_signal_off, index=index)


def run(df: pd.DataFrame, signals: pd.Series, *, stop_atr: float = 2.0,
        rr: float = 2.0, fee_bps: float = 10.0, slippage_bps: float = 5.0,
        max_hold: int = 30, exit_on_signal_off: bool = True,
        interval: str = "1d") -> BacktestResult:
    """Backtest a long/flat signal series on an OHLCV frame. Every trade is
    executed by simulate_trade() - the same function the replay and paper
    ledger use.

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

    res = BacktestResult(interval=interval)
    eq = np.ones(n)
    cur = 1.0                 # current portfolio equity
    i = 0
    while i < n:
        if i < n - 1 and sig[i] >= 1.0 and not np.isnan(atrv[i]):
            sim = simulate_trade(o, h, lo, c, i, atrv[i], stop_atr=stop_atr,
                                 rr=rr, fee_bps=fee_bps,
                                 slippage_bps=slippage_bps, max_hold=max_hold,
                                 sig=sig,
                                 exit_on_signal_off=exit_on_signal_off,
                                 index=idx)
            if sim is None:
                eq[i] = cur
                i += 1
                continue
            t = sim.trade
            res.trades.append(t)
            entry_eq = cur * (1 - fee)        # entry fee hits equity at entry
            for j in range(sim.entry_bar, sim.exit_bar):
                eq[j] = entry_eq * (c[j] / t.entry)   # mark to market
            eq[sim.exit_bar] = entry_eq * (t.exit / t.entry) * (1 - fee)
            cur = eq[sim.exit_bar]
            i = sim.exit_bar + 1
        else:
            eq[i] = cur
            i += 1

    res.equity = pd.Series(eq, index=idx)
    return res
