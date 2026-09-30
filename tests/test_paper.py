import numpy as np
import pandas as pd

from alphacouncil import paper


def _frame(closes, start="2026-06-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = closes - 0.3
    return pd.DataFrame({"open": opens, "high": closes + 1.0,
                         "low": closes - 1.0, "close": closes,
                         "volume": np.full(len(closes), 100.0)}, index=idx)


def _row(signal_date="2026-06-10", risk=4.0, rr=2.0):
    return {"id": f"{signal_date}:BTCUSD:trend_following",
            "opened_utc": f"{signal_date}T00:00:00+00:00",
            "symbol": "BTCUSD", "interval": "1d",
            "strategy": "trend_following", "params": {"fast": 20, "slow": 50},
            "signal_close": 100.0, "card_stop": 96.0, "card_target": 108.0,
            "atr": 2.0, "risk": risk, "rr": rr,
            "evidence_total": 50.0, "regime": "trending",
            "thesis": "", "bear_case": "", "risk_notes": "",
            "source": "quant-only", "status": "NEW",
            "entry": None, "entry_date": None, "stop": None, "target": None,
            "exit": None, "exit_date": None, "exit_reason": None,
            "return_pct": None, "mae_pct": None, "mfe_pct": None,
            "bars_held": 0}


BT = {"stop_atr": 2.0, "rr": 2.0, "fee_bps": 0.0, "slippage_bps": 0.0}


def test_new_activates_on_next_bar_open():
    df = _frame([100.0] * 10 + [101.0, 102.0])  # signal date = 2026-06-10
    ledger, _ = paper.resolve([_row()], {"BTCUSD": df}, bt_kwargs=BT)
    r = ledger[0]
    assert r["status"] == "ACTIVE"
    assert r["entry"] == df["open"].iloc[10]  # next bar's open
    assert r["entry_date"] == "2026-06-11"
    assert r["stop"] == r["entry"] - 4.0
    assert r["target"] == r["entry"] + 8.0


def test_new_stays_new_without_future_bar():
    df = _frame([100.0] * 10)  # last bar IS the signal date
    ledger, resolved = paper.resolve([_row()], {"BTCUSD": df}, bt_kwargs=BT)
    assert ledger[0]["status"] == "NEW" and resolved == 0


def test_active_resolves_to_target():
    closes = [100.0] * 10 + [101.0, 103.0, 106.0, 120.0]
    df = _frame(closes)
    ledger, resolved = paper.resolve([_row()], {"BTCUSD": df}, bt_kwargs=BT)
    r = ledger[0]
    assert r["status"] == "TARGET" and resolved == 1
    assert r["exit_reason"] == "take_profit"
    assert r["return_pct"] > 0
    assert r["exit"] == r["target"]


def test_active_resolves_to_stop():
    closes = [100.0] * 10 + [101.0, 99.0, 90.0]
    df = _frame(closes)
    ledger, resolved = paper.resolve([_row()], {"BTCUSD": df}, bt_kwargs=BT)
    r = ledger[0]
    assert r["status"] == "STOP"
    assert r["return_pct"] < 0


def test_resolution_is_stable_when_re_run():
    closes = [100.0] * 10 + [101.0, 99.0, 90.0]
    df = _frame(closes)
    ledger, _ = paper.resolve([_row()], {"BTCUSD": df}, bt_kwargs=BT)
    before = dict(ledger[0])
    ledger, resolved = paper.resolve(ledger, {"BTCUSD": df}, bt_kwargs=BT)
    assert resolved == 0  # closed rows are not re-resolved
    assert ledger[0] == before


def test_merge_dedupes_open_rows():
    snaps = [_row()]
    ledger, added1 = paper.merge_new([], snaps)
    ledger, added2 = paper.merge_new(ledger, [_row()])
    assert added1 == 1 and added2 == 0 and len(ledger) == 1


def test_ledger_roundtrip(tmp_path):
    p = tmp_path / "ledger.json"
    paper.save_ledger([_row()], p)
    rows = paper.load_ledger(p)
    assert len(rows) == 1 and rows[0]["status"] == "NEW"
    s = paper.ledger_summary(rows)
    assert s["open"] == 1 and s["closed"] == 0
