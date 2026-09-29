"""Deterministic evidence scoring - never an LLM "confidence: 87%".

Every component is computed from code-visible quantities and exposed
individually, so the score is auditable instead of a vibe.
"""

from __future__ import annotations

WEIGHTS = {"oos_robustness": 0.25, "trade_count": 0.15, "profit_factor": 0.15,
           "drawdown": 0.15, "regime_consistency": 0.10,
           "param_stability": 0.10, "liquidity": 0.10}


def evidence_score(metrics: dict, regime_stats: dict, current_regime: str,
                   median_dollar_volume: float) -> dict:
    pf = metrics.get("profit_factor", 0)
    pf_c = min(pf if pf != float("inf") else 99.0, 3.0) / 3.0
    dd = metrics.get("max_drawdown", 1.0)
    reg = regime_stats.get(current_regime) if regime_stats else None
    if reg and reg["trades"] >= 3:
        rpf = reg["profit_factor"]
        regime_cons = min(rpf if rpf != float("inf") else 99.0, 3.0) / 3.0
    else:
        regime_cons = 0.25  # thin evidence in this regime
    comp = {
        "oos_robustness": metrics.get("profitable_windows", 0.0),
        "trade_count": min(metrics.get("trades", 0) / 20.0, 1.0),
        "profit_factor": pf_c,
        "drawdown": 1.0 - min(dd, 0.6) / 0.6,
        "regime_consistency": regime_cons,
        "param_stability": metrics.get("param_stability", 0.0),
        "liquidity": min(median_dollar_volume / 5e7, 1.0),
    }
    total = sum(comp[k] * w for k, w in WEIGHTS.items())
    return {"total": round(100 * total, 1),
            "components": {k: round(v, 3) for k, v in comp.items()},
            "weights": WEIGHTS}
