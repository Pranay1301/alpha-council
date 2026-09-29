"""Render desk results as a markdown research report."""

from __future__ import annotations

from datetime import datetime, timezone

from .desk import DeskResult
from .disclaimers import FULL, SHORT


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def render_markdown(res: DeskResult) -> str:
    lines = []
    lines.append("# alpha-council research report")
    lines.append(f"*{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} - "
                 f"{', '.join(res.symbols)} ({res.interval}) - "
                 f"{'live council' if res.live else 'quant-only / mock council'}*\n")
    lines.append(f"> {SHORT}\n")
    if res.market_view:
        lines.append("## Market view (analyst agent)\n")
        lines.append(res.market_view + "\n")

    approved = [c for c in res.candidates if not c.vetoed]
    vetoed = [c for c in res.candidates if c.vetoed]

    lines.append(f"## Trade suggestions ({len(approved)})\n")
    if not approved:
        lines.append("No live setups passed the desk's risk policy right now. "
                     "No suggestion IS the suggestion.\n")
    for c in approved:
        m = c.metrics
        rr = (c.target - c.entry) / max(c.entry - c.stop, 1e-12)
        lines.append(f"### {c.symbol} - LONG - {c.strategy}")
        lines.append(f"- entry ~ **{c.entry:,.4f}** | stop **{c.stop:,.4f}** | "
                     f"target **{c.target:,.4f}** (R:R {rr:.1f}, ATR-based stop)")
        lines.append(f"- backtest (out-of-sample): {m.get('trades', 0)} trades, "
                     f"win rate {_fmt_pct(m.get('win_rate', 0))}, "
                     f"profit factor {m.get('profit_factor', 0):.2f}, "
                     f"max drawdown {_fmt_pct(m.get('max_drawdown', 0))}, "
                     f"Sharpe {m.get('sharpe', 0):.2f}")
        if c.thesis:
            lines.append(f"- thesis (strategist agent): {c.thesis}")
        if c.bear_case:
            lines.append(f"- bear case (critic agent): {c.bear_case}")
        lines.append("")

    if vetoed:
        lines.append(f"## Vetoed by risk policy ({len(vetoed)})\n")
        for c in vetoed:
            lines.append(f"- **{c.symbol} {c.strategy}**: {c.vetoed}")
        lines.append("")

    if res.risk_notes:
        lines.append("## Risk notes (risk manager agent)\n")
        lines.append(res.risk_notes + "\n")
    if res.errors:
        lines.append("## Pipeline warnings\n")
        for e in res.errors:
            lines.append(f"- {e}")
        lines.append("")
    lines.append("---")
    lines.append(FULL)
    return "\n".join(lines)
