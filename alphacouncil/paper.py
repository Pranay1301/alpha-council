"""Persistent paper-trading ledger.

Every approved setup is snapshotted AT SIGNAL TIME: symbol, strategy,
parameters, evidence score, regime, council output, signal bar, close and
ATR. The historical signal is never recalculated later - resolution only
walks NEW bars against the stored levels using the canonical monitoring
loop (backtest.monitor_trade), so the ledger and the backtester can never
diverge.

Lifecycle: NEW -> ACTIVE -> TARGET / STOP / EXPIRED (max hold).

Storage: a JSON file (default data/paper_ledger.json next to the repo).
That is the honest free option, with a real limit: on free Streamlit Cloud
hosting the filesystem is ephemeral, so the ledger resets on redeploy or
restart. Download the JSON as a backup, or self-host, for a durable record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import monitor_trade
from .desk import BT_KW

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_ledger.json"

STATUS_OPEN = {"NEW", "ACTIVE"}
_EXIT_TO_STATUS = {"take_profit": "TARGET", "stop_loss": "STOP",
                   "max_hold": "EXPIRED"}


def load_ledger(path: Path | str = LEDGER_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - a corrupt ledger must not kill the app
        return []
    return rows if isinstance(rows, list) else []


def save_ledger(rows: list[dict], path: Path | str = LEDGER_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, sort_keys=True))


def _row_id(symbol: str, strategy: str, signal_date: str) -> str:
    return f"{signal_date}:{symbol}:{strategy}"


def snapshot_candidates(candidates, *, source: str, interval: str,
                        signal_date: str, risk_notes: str = "") -> list[dict]:
    """Freeze each approved candidate at signal time. Card entry/stop/target
    are reference levels from the signal close; the EXECUTED entry is set
    later, at activation, from the next bar's actual open + slippage."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for c in candidates:
        if c.vetoed:
            continue
        rows.append({
            "id": _row_id(c.symbol, c.strategy, signal_date),
            "opened_utc": now,
            "symbol": c.symbol,
            "interval": interval,
            "strategy": c.strategy,
            "params": c.params,
            "signal_date": signal_date,
            "signal_close": float(c.entry),
            "card_stop": float(c.stop),
            "card_target": float(c.target),
            "atr": float(c.atr),
            "risk": float(BT_KW["stop_atr"] * c.atr),
            "rr": float(BT_KW["rr"]),
            "evidence_total": c.evidence.get("total"),
            "regime": c.regime,
            "thesis": c.thesis,
            "bear_case": c.bear_case,
            "risk_notes": risk_notes,
            "source": source,
            "status": "NEW",
            "entry": None, "entry_date": None,
            "stop": None, "target": None,
            "exit": None, "exit_date": None, "exit_reason": None,
            "return_pct": None, "mae_pct": None, "mfe_pct": None,
            "bars_held": 0,
        })
    return rows


def merge_new(ledger: list[dict], snapshots: list[dict]) -> tuple[list[dict], int]:
    """Append snapshots whose (date, symbol, strategy) is not already open."""
    open_ids = {r["id"] for r in ledger if r.get("status") in STATUS_OPEN}
    all_ids = {r["id"] for r in ledger}
    added = 0
    for s in snapshots:
        if s["id"] in open_ids or s["id"] in all_ids:
            continue
        ledger.append(s)
        added += 1
    return ledger, added


def resolve(ledger: list[dict], frames: dict[str, pd.DataFrame],
            bt_kwargs: dict | None = None) -> tuple[list[dict], int]:
    """Walk open rows against the latest bars with the canonical monitor.

    NEW rows activate at the first bar after the signal bar (entry = that
    bar's open + slippage; levels derived from the stored risk). ACTIVE rows
    are re-monitored from their entry bar over the full frame: deterministic
    and identical to the backtester, at the cost of a cheap re-walk.
    Returns (ledger, number newly resolved to a final status).
    """
    kw = dict(BT_KW if bt_kwargs is None else bt_kwargs)
    slip = kw["slippage_bps"] / 1e4
    resolved = 0
    for row in ledger:
        if row.get("status") not in STATUS_OPEN:
            continue
        df = frames.get(row["symbol"])
        if df is None or not len(df):
            continue
        h = df["high"].to_numpy()
        lo = df["low"].to_numpy()
        c = df["close"].to_numpy()
        o = df["open"].to_numpy()
        if row["status"] == "NEW":
            # signal bar = last bar ON the signal date; enter on the next one
            sig_day = row["id"].split(":", 1)[0]
            sig_bars = df.index[df.index.strftime("%Y-%m-%d") <= sig_day]
            if not len(sig_bars) or df.index.get_loc(sig_bars[-1]) + 1 >= len(df):
                continue  # no next bar yet: still NEW
            entry_bar = df.index.get_loc(sig_bars[-1]) + 1
            entry = float(o[entry_bar]) * (1 + slip)
            row.update(status="ACTIVE", entry=round(entry, 8),
                       entry_date=str(df.index[entry_bar])[:10],
                       stop=round(entry - row["risk"], 8),
                       target=round(entry + row["rr"] * row["risk"], 8))
        entry_ts = df.index[df.index.strftime("%Y-%m-%d") == row["entry_date"]]
        if not len(entry_ts):
            continue
        entry_bar = df.index.get_loc(entry_ts[0])
        sim = monitor_trade(row["entry"], row["stop"], row["target"],
                            entry_bar, h, lo, c,
                            fee_bps=kw["fee_bps"],
                            slippage_bps=kw["slippage_bps"],
                            max_hold=kw.get("max_hold", 30),
                            sig=None, exit_on_signal_off=False,
                            index=df.index)
        t = sim.trade
        row["mae_pct"] = round(float(t.mae) * 100, 2)
        row["mfe_pct"] = round(float(t.mfe) * 100, 2)
        row["bars_held"] = t.bars_held
        if t.exit_reason in _EXIT_TO_STATUS:
            row.update(status=_EXIT_TO_STATUS[t.exit_reason],
                       exit=round(float(t.exit), 8),
                       exit_date=str(t.exit_time)[:10],
                       exit_reason=t.exit_reason,
                       return_pct=round(float(t.pnl_pct) * 100, 2))
            resolved += 1
    return ledger, resolved


def ledger_summary(rows: list[dict]) -> dict:
    closed = [r for r in rows if r.get("status") in {"TARGET", "STOP", "EXPIRED"}]
    wins = [r for r in closed if (r.get("return_pct") or 0) > 0]
    pos = sum(max(0.0, r.get("return_pct") or 0.0) for r in closed)
    neg = -sum(min(0.0, r.get("return_pct") or 0.0) for r in closed)
    return {"open": len([r for r in rows if r.get("status") in STATUS_OPEN]),
            "closed": len(closed), "wins": len(wins),
            "win_rate": (len(wins) / len(closed)) if closed else 0.0,
            "profit_factor": round(pos / neg, 2) if neg else None}
