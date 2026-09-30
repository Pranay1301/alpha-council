from alphacouncil.effectiveness import paired_replay
from alphacouncil.model import ModelResponse


class VetoAllModel:
    """Council that vetoes every candidate: the council arm must go empty."""

    def complete(self, messages):
        import json
        payload = json.loads(messages[1]["content"])
        n = len(payload.get("candidates", []))
        return ModelResponse(json.dumps(
            {"vetoes": {str(i): "test veto" for i in range(n)},
             "approved": [], "notes": ""}))


def test_paired_replay_offline_runs():
    r = paired_replay(["BTCUSD"], "1d", offline=True, checkpoints=1, spacing=30)
    for k in ("quant_only", "quant_council", "veto_precision", "veto_recall",
              "note", "limits"):
        assert k in r
    assert r["model_attached"] is False
    # with no model the arms are identical by construction
    assert r["quant_only"] == r["quant_council"]
    for arm in (r["quant_only"], r["quant_council"]):
        for k in ("approved", "graded", "wins", "win_rate", "avg_return_pct",
                  "profit_factor", "worst_trade_pct", "false_positive_rate"):
            assert k in arm, k


def test_council_arm_respects_llm_vetoes():
    r = paired_replay(["BTCUSD", "ETHUSD", "SOLUSD"], "1d", offline=True,
                      model=VetoAllModel(), checkpoints=2)
    assert r["model_attached"] is True
    assert r["quant_council"]["approved"] == 0
    assert r["council_vetoes"] == r["quant_only"]["approved"]
