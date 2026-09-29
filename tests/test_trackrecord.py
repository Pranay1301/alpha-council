import pandas as pd

from alphacouncil.trackrecord import grade_call, replay, summary


def _future(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_grade_target_hit():
    fut = _future([(100, 106, 99, 105, 1), (105, 112, 104, 111, 1)])
    g = grade_call(100, 95, 110, fut)
    assert g["outcome"] == "target hit"
    assert abs(g["return_pct"] - 0.10) < 1e-9


def test_grade_stopped():
    fut = _future([(100, 101, 94, 95, 1)])
    g = grade_call(100, 95, 110, fut)
    assert g["outcome"] == "stopped"
    assert abs(g["return_pct"] - (-0.05)) < 1e-9


def test_grade_both_touched_counts_as_stopped():
    fut = _future([(100, 112, 94, 108, 1)])
    g = grade_call(100, 95, 110, fut)
    assert g["outcome"] == "stopped"


def test_grade_open():
    fut = _future([(100, 103, 98, 102, 1)])
    g = grade_call(100, 95, 110, fut)
    assert g["outcome"] == "open"
    assert abs(g["return_pct"] - 0.02) < 1e-9


def test_summary_math():
    rows = [{"outcome": "target hit", "return_pct": 10.0},
            {"outcome": "stopped", "return_pct": -5.0},
            {"outcome": "open", "return_pct": 1.0}]
    s = summary(rows)
    assert s["calls"] == 3 and s["graded"] == 2 and s["wins"] == 1
    assert s["win_rate"] == 0.5
    assert s["avg_return_pct"] == 2.0


def test_replay_offline_runs():
    rows = replay(["BTCUSD"], "1d", offline=True, checkpoints=1, spacing=30)
    assert isinstance(rows, list)
    for r in rows:
        assert r["outcome"] in {"target hit", "stopped", "open"}
