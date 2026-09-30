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


def test_ranked_indices_bounded_by_candidate_count():
    v = validate_llm("strategist.md", {"ranked": [0, 1, 2, 999], "thesis": {}},
                     valid_indices={0, 1, 2})
    assert v["ranked"] == [0, 1, 2]


def test_ranked_duplicates_removed():
    v = validate_llm("strategist.md", {"ranked": [1, 1, 0, 0], "thesis": {}},
                     valid_indices={0, 1})
    assert v["ranked"] == [1, 0]


def test_unknown_keys_dropped():
    v = validate_llm("strategist.md",
                     {"ranked": [0], "thesis": {}, "entry_override": 1,
                      "new_trade": {"symbol": "DOGEUSD"}},
                     valid_indices={0})
    assert set(v) == {"ranked", "thesis"}


def test_thesis_keys_outside_candidate_set_dropped():
    v = validate_llm("strategist.md",
                     {"ranked": "all", "thesis": {"0": "ok", "7": "ghost"}},
                     valid_indices={0})
    assert v["thesis"] == {"0": "ok"}


def test_prose_length_capped():
    v = validate_llm("critic.md", {"bear_case": {"0": "x" * 5000}},
                     valid_indices={0})
    assert len(v["bear_case"]["0"]) == 800
    r = validate_llm("risk.md",
                     {"vetoes": {}, "approved": [], "notes": "n" * 5000},
                     valid_indices=set())
    assert len(r["notes"]) == 1500


def test_approved_bounded_and_unique():
    r = validate_llm("risk.md",
                     {"vetoes": {}, "approved": [0, 0, 2, 99], "notes": ""},
                     valid_indices={0, 1, 2})
    assert r["approved"] == [0, 2]
