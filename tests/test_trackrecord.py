import numpy as np
import pandas as pd

from alphacouncil.backtest import run, simulate_trade
from alphacouncil.trackrecord import grade_call, replay, summary


def _frame(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _trendy(n=300, start=100.0, step=1.0, spread=0.5):
    idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    closes = np.array([start + i * step for i in range(n)])
    opens = closes - 0.3  # open differs from prior close: no accidental ties
    return pd.DataFrame({"open": opens, "high": closes + spread,
                         "low": closes - spread, "close": closes,
                         "volume": np.full(n, 100.0)}, index=idx)


def test_entry_is_next_bar_open_not_checkpoint_close():
    df = _trendy()
    g = grade_call(df, signal_bar=100, atr_value=2.0,
                   bt_kwargs={"stop_atr": 2.0, "rr": 2.0, "fee_bps": 0,
                              "slippage_bps": 0})
    # entry must be the open of the bar AFTER the signal bar, not the
    # signal bar's close
    assert g["entry"] == df["open"].iloc[101]
    assert g["entry"] != df["close"].iloc[100]


def test_entry_bar_range_cannot_trigger_exit():
    # signal bar 100; entry bar 101 dips below the stop intraday. The entry
    # bar's range predates the fill, so the trade must survive it.
    df = _trendy()
    atr_value = 2.0
    entry = df["open"].iloc[101]
    stop = entry - 2.0 * atr_value
    df.loc[df.index[101], "low"] = stop - 1.0   # below stop, must be ignored
    g = grade_call(df, signal_bar=100, atr_value=atr_value,
                   bt_kwargs={"stop_atr": 2.0, "rr": 2.0, "fee_bps": 0,
                              "slippage_bps": 0})
    assert g["outcome"] == "target hit"


def test_stop_wins_collision():
    df = _trendy(step=0.0)
    atr_value = 2.0
    i = 102  # bar after entry bar
    df.loc[df.index[i], "high"] = df["close"].iloc[100] + 100
    df.loc[df.index[i], "low"] = df["close"].iloc[100] - 100
    g = grade_call(df, signal_bar=100, atr_value=atr_value,
                   bt_kwargs={"stop_atr": 2.0, "rr": 2.0, "fee_bps": 0,
                              "slippage_bps": 0})
    assert g["outcome"] == "stopped"


def test_open_end_of_data():
    df = _trendy(step=0.0)  # flat market: neither stop nor target reached
    g = grade_call(df, signal_bar=298, atr_value=2.0,
                   bt_kwargs={"stop_atr": 2.0, "rr": 2.0, "fee_bps": 0,
                              "slippage_bps": 0, "max_hold": 30})
    assert g is None or g["outcome"] in {"max hold", "open (end of data)"}


def test_grading_matches_backtester_exactly():
    # the canonical function must give the replay and the backtester the
    # same trade for the same signal bar
    df = _trendy()
    from alphacouncil.indicators import atr
    atr_value = float(atr(df["high"], df["low"], df["close"]).iloc[100])
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100] = 1.0
    bt = run(df, sig, stop_atr=2.0, rr=2.0, fee_bps=10.0, slippage_bps=5.0,
             exit_on_signal_off=False).trades[0]
    g = grade_call(df, signal_bar=100, atr_value=atr_value,
                   bt_kwargs={"stop_atr": 2.0, "rr": 2.0, "fee_bps": 10.0,
                              "slippage_bps": 5.0})
    assert g["entry"] == bt.entry and g["exit"] == bt.exit
    assert g["exit_reason"] == bt.exit_reason
    assert abs(g["return_pct"] - bt.pnl_pct) < 1e-12
    assert g["bars"] == bt.bars_held


def test_simulate_trade_none_without_future_bar():
    df = _trendy(n=50)
    sim = simulate_trade(df["open"].to_numpy(), df["high"].to_numpy(),
                         df["low"].to_numpy(), df["close"].to_numpy(),
                         signal_bar=49, atr_value=2.0)
    assert sim is None


def test_summary_math():
    rows = [{"outcome": "target hit", "return_pct": 10.0},
            {"outcome": "stopped", "return_pct": -5.0},
            {"outcome": "open (end of data)", "return_pct": 1.0}]
    s = summary(rows)
    assert s["calls"] == 3 and s["graded"] == 2 and s["wins"] == 1
    assert s["win_rate"] == 0.5
    assert s["avg_return_pct"] == 2.5
    assert s["profit_factor"] == 2.0


def test_replay_offline_runs():
    rows = replay(["BTCUSD"], "1d", offline=True, checkpoints=1, spacing=30)
    assert isinstance(rows, list)
    for r in rows:
        assert r["outcome"] in {"target hit", "stopped", "max hold",
                                "open (end of data)"}
