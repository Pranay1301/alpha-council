"""Track record: the desk replays its own decision policy at past checkpoints.

At each checkpoint the FULL quant decision pipeline runs on data truncated
at that bar (no lookahead): candidate generation, hard risk vetoes,
evidence scoring, correlation and the distinct-opportunity count - the same
helpers the live dashboard uses, not a re-implementation. There is no
hindsight selection: the replay never picks "the best historical strategy";
it records exactly what the desk's policy would have shown that day.

Each recorded call is then graded with the SAME canonical execution as the
backtester (backtest.simulate_trade): entry at the NEXT bar's open plus
slippage, stop/target monitoring from the bar after entry, stop wins
collisions, fees on both sides, max-hold expiry. Signal-off exits are
disabled in grading so rule strategies and the ML model resolve under
identical, leakage-free rules.

This is a retrospective replay - suggestions only, not a live
paper-trading ledger and not observed performance.
"""

from __future__ import annotations

import pandas as pd

from .backtest import simulate_trade
from .data import load
from .desk import (BT_KW, attach_evidence, correlation_map, distinct_count,
                   evaluate_frames, hard_policy_vetoes)
from .indicators import atr

OUTCOME_LABELS = {"take_profit": "target hit", "stop_loss": "stopped",
                  "signal_off": "signal off", "max_hold": "max hold",
                  "end_of_data": "open (end of data)"}


def grade_call(df: pd.DataFrame, signal_bar: int, atr_value: float,
               bt_kwargs: dict | None = None) -> dict | None:
    """Grade one call fired at `signal_bar`'s close using the canonical
    execution (next-bar open entry + slippage; stop/target from the bar
    after entry). Returns None when no future bar exists to enter on."""
    kw = dict(BT_KW if bt_kwargs is None else bt_kwargs)
    sim = simulate_trade(df["open"].to_numpy(), df["high"].to_numpy(),
                         df["low"].to_numpy(), df["close"].to_numpy(),
                         signal_bar, atr_value, sig=None,
                         exit_on_signal_off=False, index=df.index, **kw)
    if sim is None:
        return None
    t = sim.trade
    return {"outcome": OUTCOME_LABELS[t.exit_reason],
            "exit_reason": t.exit_reason,
            "entry": t.entry, "stop": t.stop, "target": t.target,
            "exit": t.exit,
            "return_pct": t.pnl_pct,          # net of fees, as a fraction
            "bars": t.bars_held,
            "mae_pct": t.mae, "mfe_pct": t.mfe}


def decide_at_checkpoint(frames: dict[str, pd.DataFrame], interval: str,
                         errors: list[dict] | None = None) -> list:
    """Run the desk's exact quant decision policy on checkpoint-truncated
    frames. Returns the candidates the dashboard would have shown."""
    err_list: list = []
    candidates, _lb = evaluate_frames(frames, interval, err_list)
    hard_policy_vetoes(candidates)
    attach_evidence(candidates, frames)
    if errors is not None:
        for e in err_list:
            errors.append({"error": e})
    return [c for c in candidates if not c.vetoed]


def replay(symbols, interval: str = "1d", offline: bool = False,
           checkpoints: int = 4, spacing: int = 15) -> list[dict]:
    rows: list[dict] = []
    full: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            full[symbol] = load(symbol, interval, offline=offline)
        except Exception:  # noqa: BLE001
            continue
    for k in range(checkpoints, 0, -1):
        frames, cuts = {}, {}
        for s, df in full.items():
            cut = len(df) - k * spacing
            if cut < 200:  # walk-forward needs real history
                continue
            frames[s] = df.iloc[:cut]
            cuts[s] = cut
        if not frames:
            continue
        for c in decide_at_checkpoint(frames, interval):
            df_full = full[c.symbol]
            cut = cuts[c.symbol]
            atr_value = float(atr(df_full["high"], df_full["low"],
                                  df_full["close"]).to_numpy()[cut - 1])
            g = grade_call(df_full, cut - 1, atr_value)
            if g is None:
                continue
            rows.append({"date": str(df_full.index[cut - 1])[:10],
                         "symbol": c.symbol,
                         "strategy": c.strategy,
                         "entry": round(float(g["entry"]), 2),
                         "stop": round(float(g["stop"]), 2),
                         "target": round(float(g["target"]), 2),
                         "test_pf": round(c.metrics.get("profit_factor", 0), 2),
                         "outcome": g["outcome"],
                         "return_pct": round(float(g["return_pct"]) * 100, 2),
                         "bars_held": g["bars"],
                         "mae_pct": round(float(g["mae_pct"]) * 100, 2),
                         "mfe_pct": round(float(g["mfe_pct"]) * 100, 2)})
    return rows


def summary(rows: list[dict]) -> dict:
    """Aggregate a replay/ledger table. A 'win' is a positive net return
    (after fees); profit factor is undefined when there are no losses."""
    graded = [r for r in rows if r["outcome"] != "open (end of data)"]
    pool = graded if graded else rows
    wins = [r for r in pool if r["return_pct"] > 0]
    avg = sum(r["return_pct"] for r in pool) / len(pool) if pool else 0.0
    positive = sum(max(0.0, r["return_pct"]) for r in pool)
    negative = -sum(min(0.0, r["return_pct"]) for r in pool)
    pf = (positive / negative) if negative else None  # undefined with zero losses
    return {"calls": len(rows), "graded": len(pool), "wins": len(wins),
            "win_rate": (len(wins) / len(pool)) if pool else 0.0,
            "avg_return_pct": round(float(avg), 2),
            "profit_factor": round(float(pf), 2) if pf is not None else None}
