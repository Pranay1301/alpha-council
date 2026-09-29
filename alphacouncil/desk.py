"""The research desk: quant pipeline + agent council.

Two layers, strictly separated:
1. Quant (code, no LLM): data -> indicators -> strategies -> walk-forward
   backtests -> candidate setups with entry/stop/target and out-of-sample
   metrics. The LLM never sees raw price data and never computes numbers.
2. Council (LLM agents): analyst, strategist, risk manager, critic. They
   narrate, rank, veto, and attack the candidates - but every number on a
   final card comes from layer 1. This is how the tool stays honest:
   models write prose, code owns math.
"""

from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import run
from .data import load, validate
from .evidence import evidence_score
from .regime import classify, trades_by_regime
from .indicators import atr, ema, rsi
from .ml import ml_signal
from .model import MockModel, Model, parse_json
from .optimize import walk_forward
from .strategies import STRATEGIES

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

PARAM_GRIDS = {
    "trend_following": {"fast": [10, 20], "slow": [40, 50], "rsi_cap": [65.0, 70.0]},
    "mean_reversion": {"rsi_window": [14], "oversold": [25.0, 30.0], "window": [20]},
    "breakout": {"lookback": [15, 20], "vol_mult": [1.2, 1.5]},
}
BT_KW = {"stop_atr": 2.0, "rr": 2.0, "fee_bps": 10.0, "slippage_bps": 5.0}
MIN_TEST_PF = 1.0
MIN_TEST_TRADES = 4
MAX_TEST_DD = 0.35


@dataclass
class Candidate:
    index: int
    symbol: str
    interval: str
    strategy: str
    params: dict
    entry: float
    stop: float
    target: float
    atr: float
    metrics: dict          # out-of-sample (test window) metrics
    thesis: str = ""
    bear_case: str = ""
    vetoed: str = ""
    regime: str = ""
    regime_stats: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"index": self.index, "symbol": self.symbol,
                "strategy": self.strategy, "params": self.params,
                "entry": round(self.entry, 4), "stop": round(self.stop, 4),
                "target": round(self.target, 4),
                "risk_reward": round((self.target - self.entry) / max(self.entry - self.stop, 1e-12), 2),
                "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in self.metrics.items()}}


@dataclass
class DeskResult:
    market_view: str
    candidates: list[Candidate]
    risk_notes: str
    symbols: list[str]
    interval: str
    live: bool
    errors: list[str] = field(default_factory=list)
    leaderboard: list[dict] = field(default_factory=list)
    council_log: list[dict] = field(default_factory=list)
    regime: dict = field(default_factory=dict)
    correlation: dict = field(default_factory=dict)
    beta_to_btc: dict = field(default_factory=dict)
    distinct_opportunities: int = 0
    audit_log: list[dict] = field(default_factory=list)



def _ml_walk_forward(df: pd.DataFrame, interval: str, n_folds: int = 3,
                     train_frac: float = 0.5, bt_kwargs: dict | None = None):
    """ML through the same rolling windows as the rule strategies: refit on
    each fold's training window, signals only on its held-out window."""
    import numpy as np
    bt_kwargs = bt_kwargs or {}
    n = len(df)
    test_len = int(n * (1 - train_frac) / n_folds)
    tms = []
    for k in range(n_folds):
        split = int(n * train_frac) + k * test_len
        if split + test_len > n:
            break
        sub = df.iloc[:split + test_len]
        sig = ml_signal(sub, fit_frac=split / len(sub))
        res = run(df.iloc[split:split + test_len], sig.iloc[split:],
                  interval=interval, **bt_kwargs)
        m = res.metrics()
        if m.get("trades", 0):
            tms.append(m)
    if not tms:
        return None
    pfs = [m.get("profit_factor", 0) for m in tms]
    return {"win_rate": float(np.median([m.get("win_rate", 0) for m in tms])),
            "profit_factor": float(np.median(pfs)),
            "max_drawdown": float(max(m.get("max_drawdown", 0) for m in tms)),
            "sharpe": float(np.median([m.get("sharpe", 0) for m in tms])),
            "total_return": float(np.median([m.get("total_return", 0) for m in tms])),
            "avg_r": float(np.median([m.get("avg_r", 0) for m in tms])),
            "trades": int(sum(m.get("trades", 0) for m in tms)),
            "folds": len(tms),
            "profitable_windows": float(np.mean([pf > 1.0 for pf in pfs])),
            "param_stability": 1.0}

def _quant_candidates(symbols: list[str], interval: str, offline: bool,
                      errors: list[str]) -> tuple[list[Candidate], list[dict]]:
    candidates: list[Candidate] = []
    leaderboard: list[dict] = []
    for symbol in symbols:
        try:
            df = load(symbol, interval, offline=offline)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: data load failed: {exc}")
            continue
        try:
            for issue in validate(df, interval):
                errors.append(f"{symbol} data quality: {issue}")
            current_atr = float(atr(df["high"], df["low"], df["close"]).iloc[-1])
            price = float(df["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
            errors.append(f"{symbol}: data validation failed: {exc!r}")
            continue

        for name, fn in STRATEGIES.items():
            try:
                wf = walk_forward(df, name, fn, PARAM_GRIDS[name],
                                  interval=interval, bt_kwargs=BT_KW)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}/{name}: walk-forward failed: {exc}")
                continue
            if wf is None:
                continue
            sig_now = float(fn(df, **wf.best_params).iloc[-1])
            tm = wf.test_metrics
            leaderboard.append({
                "symbol": symbol, "strategy": name,
                "win_rate": round(tm.get("win_rate", 0), 3),
                "profit_factor": round(tm.get("profit_factor", 0), 2),
                "max_drawdown": round(tm.get("max_drawdown", 0), 3),
                "trades": tm.get("trades", 0),
                "signal_now": "live" if sig_now >= 1.0 else "flat"})
            labels = classify(df)
            full_bt = run(df, fn(df, **wf.best_params), interval=interval, **BT_KW)
            reg_stats = trades_by_regime(df, full_bt.trades)
            if sig_now < 1.0:
                continue  # only suggest setups that are live right now
            risk = BT_KW["stop_atr"] * current_atr
            candidates.append(Candidate(
                index=len(candidates), symbol=symbol, interval=interval,
                strategy=name, params=wf.best_params, entry=price,
                stop=price - risk, target=price + BT_KW["rr"] * risk,
                atr=current_atr, metrics=wf.test_metrics,
                regime=str(labels.iloc[-1]), regime_stats=reg_stats))

        # ML signal: same rolling walk-forward windows as the rule
        # strategies - refit per fold, held-out evaluation only
        try:
            ml_metrics = _ml_walk_forward(df, interval, bt_kwargs=BT_KW)
            sig = ml_signal(df)
            if ml_metrics is not None:
                leaderboard.append({
                    "symbol": symbol, "strategy": "ml_logistic",
                    "win_rate": round(ml_metrics.get("win_rate", 0), 3),
                    "profit_factor": round(ml_metrics.get("profit_factor", 0), 2),
                    "max_drawdown": round(ml_metrics.get("max_drawdown", 0), 3),
                    "trades": ml_metrics.get("trades", 0),
                    "signal_now": "live" if sig.iloc[-1] >= 1.0 else "flat"})
            if ml_metrics is not None and sig.iloc[-1] >= 1.0:
                risk = BT_KW["stop_atr"] * current_atr
                candidates.append(Candidate(
                    index=len(candidates), symbol=symbol, interval=interval,
                    strategy="ml_logistic", params={"horizon": 5, "threshold": 0.55},
                    entry=price, stop=price - risk,
                    target=price + BT_KW["rr"] * risk, atr=current_atr,
                    metrics=ml_metrics))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}/ml: failed: {exc}")
    return candidates, leaderboard


def _quant_summary(symbols: list[str], interval: str, offline: bool) -> dict:
    out = {}
    for symbol in symbols:
        try:
            df = load(symbol, interval, offline=offline)
        except Exception:  # noqa: BLE001
            continue
        close = df["close"]
        out[symbol] = {
            "price": round(float(close.iloc[-1]), 4),
            "change_7b_pct": round(float(close.iloc[-1] / close.iloc[-8] - 1) * 100, 2) if len(close) > 8 else None,
            "change_30b_pct": round(float(close.iloc[-1] / close.iloc[-31] - 1) * 100, 2) if len(close) > 31 else None,
            "rsi14": round(float(rsi(close).iloc[-1]), 1),
            "ema20_vs_ema50": "above" if ema(close, 20).iloc[-1] > ema(close, 50).iloc[-1] else "below",
            "atr_pct": round(float(atr(df["high"], df["low"], close).iloc[-1] / close.iloc[-1]) * 100, 2),
        }
    return out




def _clean_str_dict(v) -> dict:
    """Keep only int-coercible keys mapping to strings."""
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, val in v.items():
        if str(k).lstrip("-").isdigit() and isinstance(val, str):
            out[str(k)] = val
    return out


def validate_llm(prompt_name: str, parsed: dict) -> dict:
    """Strict shape checking for council responses. A malformed field is
    dropped, never passed downstream - models write prose, code owns math."""
    if not isinstance(parsed, dict):
        return {}
    if prompt_name == "analyst.md":
        return {k: v for k, v in parsed.items() if isinstance(v, str)}
    if prompt_name == "strategist.md":
        ranked = parsed.get("ranked", "all")
        if ranked != "all":
            if not isinstance(ranked, list):
                ranked = "all"
            else:
                ranked = [int(i) for i in ranked
                          if isinstance(i, (int, float, str)) and str(i).lstrip("-").isdigit() and int(i) >= 0]
        return {"ranked": ranked, "thesis": _clean_str_dict(parsed.get("thesis"))}
    if prompt_name == "risk.md":
        approved = parsed.get("approved", [])
        if not isinstance(approved, list):
            approved = []
        notes = parsed.get("notes", "")
        return {"vetoes": _clean_str_dict(parsed.get("vetoes")),
                "approved": [int(i) for i in approved
                             if isinstance(i, (int, float, str)) and str(i).lstrip("-").isdigit()],
                "notes": notes if isinstance(notes, str) else ""}
    if prompt_name == "critic.md":
        return {"bear_case": _clean_str_dict(parsed.get("bear_case"))}
    return parsed

def _ask(model: Model | None, prompt_name: str, payload: dict, fallback: dict,
         raw_as: str | None = None, audit_log: list[dict] | None = None) -> dict:
    if model is None:
        return fallback
    system = (PROMPTS_DIR / prompt_name).read_text()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    resp = model.complete(messages)
    if audit_log is not None:
        def digest(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        audit_log.append({
            "stage": prompt_name.removesuffix(".md"),
            "provider": getattr(model, "provider", type(model).__name__),
            "model": getattr(model, "model", "mock"),
            "timestamp_utc": started_at,
            "prompt_version": digest(system),
            "input_hash": digest(messages[1]["content"]),
            "output_hash": digest(resp.content),
            "latency_ms": round((time.monotonic() - start) * 1000),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        })
    parsed = parse_json(resp.content)
    if parsed:
        return validate_llm(prompt_name, parsed)
    text = resp.content.strip()
    if raw_as and text:
        return {raw_as: text[:1500]}
    return fallback


def run_desk(symbols: list[str], interval: str = "1d", *, offline: bool = False,
             model: Model | None = None) -> DeskResult:
    errors: list[str] = []
    candidates, leaderboard = _quant_candidates(symbols, interval, offline, errors)
    summary = _quant_summary(symbols, interval, offline)
    for s in symbols:
        try:
            summary[s]["regime"] = str(classify(load(s, interval, offline=offline)).iloc[-1])
        except Exception:  # noqa: BLE001
            pass
    summary["interval"] = interval
    summary["live_candidates"] = len(candidates)

    audit_log: list[dict] = []
    analyst = _ask(model, "analyst.md", summary,
                   {"market_view": ""}, raw_as="market_view", audit_log=audit_log)
    market_view = analyst.get("market_view") or analyst.get("reason") or ""
    if not market_view and model is None:
        market_view = "(no model attached - quant-only run)"
    elif not market_view:
        market_view = next((v for v in analyst.values() if isinstance(v, str)), "")

    payload = {"candidates": [c.to_json() for c in candidates]}
    strat = _ask(model, "strategist.md", payload, {"ranked": "all", "thesis": {}},
                audit_log=audit_log)
    # thesis keys refer to the ORIGINAL indices in payload - apply before reordering
    for k, v in (strat.get("thesis") or {}).items():
        if str(k).isdigit() and int(k) in {c.index for c in candidates}:
            by_index_all = {c.index: c for c in candidates}
            by_index_all[int(k)].thesis = v
    ranked = strat.get("ranked", "all")
    order = ([c.index for c in candidates] if ranked == "all"
             else [int(i) for i in ranked if int(i) < len(candidates)])
    by_index = {c.index: c for c in candidates}
    candidates = [by_index[i] for i in order if i in by_index]
    for i, c in enumerate(candidates):
        c.index = i  # reindex in ranked order; payloads below use these

    risk_payload = {"candidates": [c.to_json() for c in candidates]}
    risk = _ask(model, "risk.md", risk_payload,
                {"vetoes": {}, "approved": [c.index for c in candidates], "notes": ""},
                audit_log=audit_log)
    # code enforces the risk policy too - the LLM's veto is advisory,
    # the hard rules below are not
    for c in candidates:
        m = c.metrics
        reasons = []
        if m.get("profit_factor", 0) < MIN_TEST_PF:
            reasons.append(f"test profit factor {m.get('profit_factor', 0):.2f} < {MIN_TEST_PF}")
        if m.get("trades", 0) < MIN_TEST_TRADES:
            reasons.append(f"only {m.get('trades', 0)} test trades")
        if m.get("max_drawdown", 1) > MAX_TEST_DD:
            reasons.append(f"test max drawdown {m.get('max_drawdown', 0):.0%} > {MAX_TEST_DD:.0%}")
        if reasons:
            c.vetoed = "; ".join(reasons)
    for k, v in (risk.get("vetoes") or {}).items():
        if str(k).isdigit() and int(k) < len(candidates) and not candidates[int(k)].vetoed:
            candidates[int(k)].vetoed = f"risk manager: {v}"

    approved = [c for c in candidates if not c.vetoed]
    critic = _ask(model, "critic.md",
                  {"candidates": [c.to_json() for c in approved]},
                  {"bear_case": {}}, audit_log=audit_log)
    approved_idx = {c.index: c for c in approved}
    for k, v in (critic.get("bear_case") or {}).items():
        if str(k).isdigit() and int(k) in approved_idx:
            approved_idx[int(k)].bear_case = v

    # portfolio layer: pairwise return correlation + distinct opportunities
    corr: dict = {}
    rets = {}
    for s in symbols:
        try:
            rets[s] = load(s, interval, offline=offline)["close"].pct_change().dropna().tail(180)
        except Exception:  # noqa: BLE001
            pass
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            if a in rets and b in rets:
                joined = pd.concat([rets[a], rets[b]], axis=1, join="inner")
                if len(joined) > 30:
                    corr[f"{a}/{b}"] = round(float(joined.corr().iloc[0, 1]), 3)
    # Empirical beta, not an investable factor model: aligned close-to-close
    # returns over up to 180 bars. No BTC data means beta is unavailable.
    beta_to_btc: dict = {}
    if "BTCUSD" in rets and len(rets["BTCUSD"]) > 30:
        btc = rets["BTCUSD"]
        for sym, returns in rets.items():
            aligned = pd.concat([returns, btc], axis=1, join="inner").dropna()
            if len(aligned) < 30:
                continue
            variance = float(aligned.iloc[:, 1].var())
            if variance > 1e-12:
                beta_to_btc[sym] = round(float(aligned.iloc[:, 0].cov(
                    aligned.iloc[:, 1]) / variance), 2)
    approved_list = [c for c in candidates if not c.vetoed]
    for c in approved_list:
        try:
            df_c = load(c.symbol, interval, offline=offline)
            med_dv = float((df_c["close"] * df_c["volume"]).tail(90).median())
        except Exception:  # noqa: BLE001
            med_dv = 0.0
        c.evidence = evidence_score(c.metrics, c.regime_stats, c.regime, med_dv)
    distinct = 0
    picked: list[str] = []
    for c in sorted(approved_list, key=lambda x: -x.evidence.get("total", 0)):
        dup = False
        for p in picked:
            if p == c.symbol or abs(corr.get(f"{p}/{c.symbol}", corr.get(f"{c.symbol}/{p}", 1.0))) >= 0.7:
                dup = True
                break
        if not dup:
            distinct += 1
            picked.append(c.symbol)

    council_log = [
        {"stage": "analyst - market view", "output": market_view or "(no market view)"},
        {"stage": "strategist - ranking and theses",
         "output": json.dumps(strat, indent=2)[:3000]},
        {"stage": "risk manager - vetoes and notes",
         "output": json.dumps(risk, indent=2)[:3000]},
        {"stage": "critic - bear cases",
         "output": json.dumps(critic, indent=2)[:3000]},
    ]
    return DeskResult(market_view=market_view, candidates=candidates,
                      risk_notes=risk.get("notes", ""), symbols=symbols,
                      interval=interval,
                      live=model is not None and not isinstance(model, MockModel),
                      errors=errors, leaderboard=leaderboard,
                      council_log=council_log, regime=summary,
                      correlation=corr, beta_to_btc=beta_to_btc,
                      distinct_opportunities=distinct,
                      audit_log=audit_log)
