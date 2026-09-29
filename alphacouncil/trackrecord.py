"""Track record: the desk grades its own past calls.

Replays the quant pipeline at historical checkpoints on the SAME data it
would have seen then (frames are truncated at the checkpoint, and a signal
must have been live on that exact bar - no lookahead), then grades each
suggested trade against what price actually did next. Suggestions only.
"""

from __future__ import annotations

import pandas as pd

from .data import load
from .desk import BT_KW, PARAM_GRIDS
from .indicators import atr
from .optimize import walk_forward
from .strategies import STRATEGIES


def _best_call_at(df: pd.DataFrame, symbol: str, interval: str) -> dict | None:
    price = float(df["close"].iloc[-1])
    cur_atr = float(atr(df["high"], df["low"], df["close"]).iloc[-1])
    best = None
    for name, fn in STRATEGIES.items():
        try:
            wf = walk_forward(df, name, fn, PARAM_GRIDS[name],
                              interval=interval, bt_kwargs=BT_KW)
        except Exception:  # noqa: BLE001
            continue
        if wf is None:
            continue
        sig_now = float(fn(df, **wf.best_params).iloc[-1])
        if sig_now < 1.0:
            continue
        pf = wf.test_metrics.get("profit_factor", 0)
        if best is None or pf > best["test_pf"]:
            risk = BT_KW["stop_atr"] * cur_atr
            best = {"strategy": name, "entry": price, "stop": price - risk,
                    "target": price + BT_KW["rr"] * risk, "test_pf": pf}
    return best


def grade_call(entry: float, stop: float, target: float,
               future: pd.DataFrame) -> dict:
    """Grade a long call against the bars that followed it. A bar touching
    both stop and target counts as stopped out (conservative)."""
    for i in range(len(future)):
        lo = float(future["low"].iloc[i])
        hi = float(future["high"].iloc[i])
        if lo <= stop:
            return {"outcome": "stopped", "return_pct": (stop - entry) / entry,
                    "bars": i + 1}
        if hi >= target:
            return {"outcome": "target hit",
                    "return_pct": (target - entry) / entry, "bars": i + 1}
    last = float(future["close"].iloc[-1]) if len(future) else entry
    return {"outcome": "open", "return_pct": (last - entry) / entry,
            "bars": len(future)}


def replay(symbols, interval: str = "1d", offline: bool = False,
           checkpoints: int = 4, spacing: int = 15) -> list[dict]:
    rows: list[dict] = []
    for symbol in symbols:
        try:
            df = load(symbol, interval, offline=offline)
        except Exception:  # noqa: BLE001
            continue
        n = len(df)
        for k in range(checkpoints, 0, -1):
            cut = n - k * spacing
            if cut < 200:  # walk-forward needs real history
                continue
            past, future = df.iloc[:cut], df.iloc[cut:]
            call = _best_call_at(past, symbol, interval)
            if call is None:
                continue
            g = grade_call(call["entry"], call["stop"], call["target"], future)
            rows.append({"date": str(past.index[-1])[:10], "symbol": symbol,
                         "strategy": call["strategy"],
                         "entry": round(call["entry"], 2),
                         "stop": round(call["stop"], 2),
                         "target": round(call["target"], 2),
                         "test_pf": round(call["test_pf"], 2),
                         "outcome": g["outcome"],
                         "return_pct": round(g["return_pct"] * 100, 2)})
    return rows


def summary(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["outcome"] != "open"]
    wins = [r for r in graded if r["outcome"] == "target hit"]
    avg = sum(r["return_pct"] for r in rows) / len(rows) if rows else 0.0
    return {"calls": len(rows), "graded": len(graded), "wins": len(wins),
            "win_rate": (len(wins) / len(graded)) if graded else 0.0,
            "avg_return_pct": round(avg, 2)}
