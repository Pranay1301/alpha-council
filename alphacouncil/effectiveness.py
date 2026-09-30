"""Council effectiveness: paired quant-only vs quant+council replay.

At each historical checkpoint the same truncated frames feed ONE candidate
set (the desk's exact quant pipeline). That set is then decided twice:

    QUANT-ONLY    = candidates passing the hard risk policy
    QUANT+COUNCIL = the same candidates, minus the council's advisory vetoes
                    (risk-stage LLM output, validated against the live
                    candidate set)

Both arms face the same future bars and are graded by the same canonical
simulate_trade execution. Only the risk stage changes set membership -
the analyst narrates, the strategist reorders, the critic annotates - so
those stages are not replayed and this is stated in the output.

Without a live model attached, the council arm is identical to the quant
arm by construction; the output says so instead of manufacturing a delta.
Metrics on small samples are descriptive, not proof.
"""

from __future__ import annotations

import pandas as pd

from .data import load
from .desk import (Candidate, _ask, attach_evidence, evaluate_frames,
                   hard_policy_vetoes)
from .indicators import atr
from .model import Model
from .trackrecord import OUTCOME_LABELS, grade_call


def _grade(df: pd.DataFrame, cut: int, c: Candidate) -> dict:
    atr_value = float(atr(df["high"], df["low"], df["close"]).to_numpy()[cut - 1])
    g = grade_call(df, cut - 1, atr_value)
    row = {"date": str(df.index[cut - 1])[:10], "symbol": c.symbol,
           "strategy": c.strategy,
           "test_pf": round(float(c.metrics.get("profit_factor", 0)), 2)}
    if g is None:
        row.update(outcome="no future bar", return_pct=None)
        return row
    row.update(outcome=OUTCOME_LABELS[g["exit_reason"]],
               return_pct=round(float(g["return_pct"]) * 100, 2),
               bars_held=g["bars"])
    return row


def _arm_metrics(rows: list[dict]) -> dict:
    graded = [r for r in rows if r.get("return_pct") is not None]
    wins = [r for r in graded if r["return_pct"] > 0]
    pos = sum(max(0.0, r["return_pct"]) for r in graded)
    neg = -sum(min(0.0, r["return_pct"]) for r in graded)
    return {"approved": len(rows), "graded": len(graded), "wins": len(wins),
            "win_rate": (len(wins) / len(graded)) if graded else 0.0,
            "avg_return_pct": round(sum(r["return_pct"] for r in graded) / len(graded), 2) if graded else 0.0,
            "profit_factor": round(pos / neg, 2) if neg else None,
            "worst_trade_pct": round(min((r["return_pct"] for r in graded), default=0.0), 2),
            "false_positive_rate": (len([r for r in graded if r["return_pct"] <= 0]) / len(graded)) if graded else 0.0}


def paired_replay(symbols, interval: str = "1d", offline: bool = False,
                  model: Model | None = None, checkpoints: int = 4,
                  spacing: int = 15) -> dict:
    full: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            full[symbol] = load(symbol, interval, offline=offline)
        except Exception:  # noqa: BLE001
            continue
    quant_rows: list[dict] = []
    council_rows: list[dict] = []
    vetoed_rows: list[dict] = []
    generated = rejected = 0
    for k in range(checkpoints, 0, -1):
        frames, cuts = {}, {}
        for s, df in full.items():
            cut = len(df) - k * spacing
            if cut < 200:
                continue
            frames[s] = df.iloc[:cut]
            cuts[s] = cut
        if not frames:
            continue
        errors: list = []
        candidates, _lb = evaluate_frames(frames, interval, errors)
        generated += len(candidates)
        hard_policy_vetoes(candidates)
        attach_evidence(candidates, frames)
        quant_ok = {id(c) for c in candidates if not c.vetoed}
        rejected += len(candidates) - len(quant_ok)
        if model is not None:
            for i, c in enumerate(candidates):
                c.index = i  # indices the council sees for this checkpoint
            risk = _ask(model, "risk.md",
                        {"candidates": [c.to_json() for c in candidates]},
                        {"vetoes": {}, "approved": [], "notes": ""},
                        valid_indices=set(range(len(candidates))))
            for key, v in (risk.get("vetoes") or {}).items():
                if str(key).isdigit():
                    i = int(key)
                    if i < len(candidates) and not candidates[i].vetoed:
                        candidates[i].vetoed = f"risk manager: {v}"
        for c in candidates:
            row = _grade(full[c.symbol], cuts[c.symbol], c)
            if id(c) in quant_ok:
                quant_rows.append(row)
                if c.vetoed:  # council vetoed what the quant arm approved
                    vetoed_rows.append(row)
                else:
                    council_rows.append(row)
    qm = _arm_metrics(quant_rows)
    cm = _arm_metrics(council_rows)
    veto_losers = [r for r in vetoed_rows
                   if r.get("return_pct") is not None and r["return_pct"] <= 0]
    quant_losers = [r for r in quant_rows
                    if r.get("return_pct") is not None and r["return_pct"] <= 0]
    return {
        "checkpoints": checkpoints, "generated": generated,
        "quant_rejected": rejected,
        "council_vetoes": len(vetoed_rows),
        "quant_only": qm,
        "quant_council": cm,
        "veto_precision": (len(veto_losers) / len(vetoed_rows)) if vetoed_rows else None,
        "veto_recall": (len(veto_losers) / len(quant_losers)) if quant_losers else None,
        "model_attached": model is not None,
        "note": ("council arm decided by a live model at each checkpoint"
                 if model is not None else
                 "no live model attached: the council arm equals the quant "
                 "arm by construction - run with a live council for a real "
                 "comparison"),
        "limits": ("small retrospective sample; descriptive, not proof. "
                   "Only the risk stage changes membership, so analyst, "
                   "strategist and critic stages are not replayed."),
        "quant_rows": quant_rows, "council_rows": council_rows,
        "vetoed_rows": vetoed_rows,
    }
