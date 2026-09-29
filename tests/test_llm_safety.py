import json

from alphacouncil.desk import run_desk, validate_llm
from alphacouncil.model import ModelResponse


class EvilModel:
    """Returns numbers that disagree with the quant layer and junk shapes."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        prompt = messages[0]["content"]
        if "analyst" in prompt:
            return ModelResponse(json.dumps({"market_view": "buy everything", "ranked": ["banana", 999, -4]}))
        if "strategist" in prompt:
            return ModelResponse(json.dumps({"ranked": ["banana", 999, -4, 0],
                                             "thesis": {"0": "entry is 101 and target is 1,000,000"}}))
        if "risk" in prompt:
            return ModelResponse(json.dumps({"vetoes": {}, "approved": "yes", "notes": 42}))
        return ModelResponse(json.dumps({"bear_case": {"0": "scam"}}))


def test_schema_validation_drops_junk():
    v = validate_llm("strategist.md", {"ranked": ["banana", 999, -4], "thesis": {0: "ok", "x": 1}})
    assert v["ranked"] == [999]
    assert v["thesis"] == {"0": "ok"}
    r = validate_llm("risk.md", {"vetoes": [], "approved": "yes", "notes": 42})
    assert r["approved"] == [] and r["notes"] == ""


def test_council_never_creates_or_changes_trade_numbers():
    quant = run_desk(["BTCUSD"], "1d", offline=True, model=None)
    evil = run_desk(["BTCUSD"], "1d", offline=True, model=EvilModel())
    q = [(c.symbol, c.strategy, c.entry, c.stop, c.target) for c in quant.candidates]
    e = [(c.symbol, c.strategy, c.entry, c.stop, c.target) for c in evil.candidates]
    assert q == e  # the LLM cannot alter a single number or add a trade
