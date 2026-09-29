"""End-to-end desk tests (offline, mock council). Run: python tests/test_desk.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from alphacouncil.desk import run_desk
from alphacouncil.report import render_markdown
from run_desk import mock_council


def test_desk_runs_offline_on_bundled_data():
    res = run_desk(["BTCUSD", "ETHUSD", "SOLUSD"], "1d",
                   offline=True, model=mock_council())
    assert res.market_view
    assert not res.live  # mock council is not a live council
    for c in res.candidates:
        assert c.stop < c.entry < c.target
        assert c.metrics.get("trades", 0) >= 0


def test_risk_policy_vetoes_losing_strategies():
    res = run_desk(["BTCUSD", "ETHUSD", "SOLUSD"], "1d",
                   offline=True, model=mock_council())
    for c in res.candidates:
        if not c.vetoed:
            m = c.metrics
            assert m.get("profit_factor", 0) >= 1.0
            assert m.get("trades", 0) >= 4
            assert m.get("max_drawdown", 1) <= 0.35


def test_report_has_disclaimers_and_numbers():
    res = run_desk(["BTCUSD"], "1d", offline=True, model=mock_council())
    md = render_markdown(res)
    assert "not financial advice" in md
    assert "DISCLAIMER" in md
    assert "BTCUSD" in md


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"pass: {name}")
    print("all desk tests passed")
