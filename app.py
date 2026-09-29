"""alpha-council dashboard (Streamlit).

    pip install -r requirements.txt
    streamlit run app.py

Runs offline out of the box (bundled data + mock council). Add a free-tier
API key in the sidebar for a live LLM council on fresh market data.

Suggestions only - not financial advice.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from alphacouncil.data import KRAKEN_PAIRS, load
from alphacouncil.desk import run_desk
from alphacouncil.disclaimers import FULL, SHORT
from alphacouncil.report import render_markdown

st.set_page_config(page_title="alpha-council", page_icon="📊", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def _track_record(symbols: tuple, interval: str, offline: bool):
    from alphacouncil.trackrecord import replay, summary
    rows = replay(list(symbols), interval, offline)
    return rows, summary(rows)

st.title("alpha-council")
st.caption("An AI research desk for crypto. LLM agents propose, rank and "
           "attack trade setups; code owns every number; risk policy can "
           "veto anything.")
st.warning(SHORT)

with st.sidebar:
    st.header("Universe")
    symbols = st.multiselect("Coins", sorted(KRAKEN_PAIRS),
                             default=["BTCUSD", "ETHUSD", "SOLUSD"])
    interval = st.selectbox("Timeframe", ["1d", "1h"])
    st.header("Council")
    live = st.toggle("Live LLM council", value=False,
                     help="Off: bundled data + scripted mock council. "
                          "On: fresh market data + a real LLM.")
    provider = st.selectbox("Free-tier provider", ["groq", "zai", "openrouter"])
    api_key = st.text_input("API key", type="password",
                            help=f"Free key from the {provider} console. "
                                 "Never stored.")
    run = st.button("Run the desk", type="primary", use_container_width=True)

if run and symbols:
    model = None
    if live:
        import os
        from alphacouncil.model import FREE_PROVIDERS, OpenAIModel
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                api_key = ""
        if not api_key:
            st.error("Live council needs an API key (free tier is fine).")
            st.stop()
        base_url, default_model, _ = FREE_PROVIDERS[provider]
        model = OpenAIModel(default_model, base_url=base_url, api_key=api_key)

    with st.spinner("Desk is working: data -> strategies -> walk-forward "
                    "backtests -> council debate..."):
        res = run_desk(symbols, interval, offline=not live, model=model)

    st.subheader("Market view")
    st.write(res.market_view)
    if res.regime:
        bits = []
        for s, d in res.regime.items():
            if isinstance(d, dict) and "ema20_vs_ema50" in d:
                bits.append(f"**{s}**: EMA20 {d['ema20_vs_ema50']} EMA50 "
                            f"\u00b7 ATR {d.get('atr_pct', '?')}%")
        if bits:
            st.caption(" \u00b7 ".join(bits))

    st.subheader("Market overview")
    ov_rows = []
    for s in res.symbols:
        dd = res.regime.get(s, {}) if isinstance(res.regime, dict) else {}
        n_ok = len([c for c in res.candidates if c.symbol == s and not c.vetoed])
        ov_rows.append({"Coin": s, "Regime": dd.get("regime", "-"),
                        "Trend (EMA20/50)": dd.get("ema20_vs_ema50", "-"),
                        "ATR %": dd.get("atr_pct", "-"),
                        "Approved setups": n_ok})
    st.dataframe(pd.DataFrame(ov_rows), use_container_width=True, hide_index=True)

    approved = [c for c in res.candidates if not c.vetoed]
    vetoed = [c for c in res.candidates if c.vetoed]

    st.subheader(f"Trade suggestions ({len(approved)})")
    if not approved:
        st.error("NO ACTIONABLE SETUPS")
        for c in vetoed:
            st.markdown(f"- **{c.symbol}** `{c.strategy}` - rejected: {c.vetoed}")
        if not vetoed:
            st.markdown("No live signals on any selected coin.")
        st.caption("The desk found no setup meeting the current research "
                   "policy. No suggestion IS the suggestion.")
    for c in approved:
        m = c.metrics
        rr = (c.target - c.entry) / max(c.entry - c.stop, 1e-12)
        with st.container(border=True):
            st.markdown(f"### {c.symbol} - LONG - `{c.strategy}`")
            cols = st.columns(4)
            cols[0].metric("Entry", f"{c.entry:,.2f}")
            cols[1].metric("Stop loss", f"{c.stop:,.2f}", delta=f"-{(c.entry-c.stop)/c.entry:.1%}",
                           delta_color="inverse")
            cols[2].metric("Take profit", f"{c.target:,.2f}",
                           delta=f"+{(c.target-c.entry)/c.entry:.1%}")
            cols[3].metric("R:R", f"{rr:.1f}")
            cols = st.columns(4)
            cols[0].metric("Test win rate", f"{m.get('win_rate', 0):.0%}",
                           help="Out-of-sample backtest window")
            cols[1].metric("Profit factor", f"{m.get('profit_factor', 0):.2f}",
                           help="Out-of-sample: gross wins / gross losses")
            cols[2].metric("Max drawdown", f"{m.get('max_drawdown', 0):.0%}")
            cols[3].metric("Test trades", m.get("trades", 0))
            if c.thesis:
                st.markdown(f"**Thesis (strategist):** {c.thesis}")
            if c.bear_case:
                st.markdown(f"**Bear case (critic):** {c.bear_case}")
            with st.expander("Why this setup exists"):
                dd = res.regime.get(c.symbol, {}) if isinstance(res.regime, dict) else {}
                st.markdown(f"**Signal** `{c.strategy}` \u00b7 current regime `{c.regime or '?'}`")
                st.markdown(f"**Why now**  \u2713 EMA20 {dd.get('ema20_vs_ema50', '?')} EMA50 "
                            f"\u00b7 \u2713 ATR {dd.get('atr_pct', '?')}% of price "
                            f"\u00b7 \u2713 signal live on the last bar")
                st.markdown(f"**Backtest evidence**  \u2713 {m.get('folds', '?')} rolling windows "
                            f"\u00b7 \u2713 {m.get('trades', '?')} OOS trades "
                            f"\u00b7 \u2713 median PF {m.get('profit_factor', 0):.2f} "
                            f"\u00b7 \u2713 median Sharpe {m.get('sharpe', 0):.2f} "
                            f"\u00b7 \u2713 worst DD {m.get('max_drawdown', 0):.0%} "
                            f"\u00b7 \u2713 {m.get('profitable_windows', 0):.0%} profitable windows "
                            f"\u00b7 \u2713 param stability {m.get('param_stability', 0):.0%}")
                if c.regime_stats:
                    def _pfkey(kv):
                        pf = kv[1]["profit_factor"]
                        return pf if pf != float("inf") else 99.0
                    best_reg = max(c.regime_stats.items(), key=_pfkey)
                    st.markdown(f"**Regime fit**  past edge concentrated in "
                                f"`{best_reg[0]}` (PF {best_reg[1]['profit_factor']:.2f} "
                                f"over {best_reg[1]['trades']} trades)")
                st.markdown("**Council**")
                st.text(f"Analyst     -> {res.market_view[:180]}")
                st.text(f"Strategist  -> {(c.thesis or 'no thesis')[:180]}")
                st.text(f"Risk        -> no veto (passed hard policy)")
                st.text(f"Critic      -> {(c.bear_case or 'no objection')[:180]}")
            try:
                df = load(c.symbol, interval, offline=not live)
                st.line_chart(df["close"].tail(120), height=180)
            except Exception:
                pass

    if vetoed:
        with st.expander(f"Vetoed by risk policy ({len(vetoed)})"):
            for c in vetoed:
                st.markdown(f"- **{c.symbol} {c.strategy}**: {c.vetoed}")
    if res.risk_notes:
        st.subheader("Risk notes")
        st.write(res.risk_notes)
    if res.errors:
        with st.expander("Pipeline warnings"):
            for e in res.errors:
                st.text(e)

    if res.leaderboard:
        st.subheader("Strategy leaderboard (out-of-sample)")
        lb = pd.DataFrame(res.leaderboard).sort_values(
            "profit_factor", ascending=False)
        st.dataframe(lb, use_container_width=True, hide_index=True)

    if res.council_log:
        with st.expander("Full council debate"):
            for stage in res.council_log:
                st.markdown(f"**{stage['stage']}**")
                st.text(stage["output"][:2000])

    st.subheader("Track record - the desk grades its own calls")
    st.caption("The quant engine replayed at past checkpoints on the data "
               "it would have seen then, graded against what price did "
               "next. Backtested replay, not live trading results.")
    with st.spinner("Grading past signals..."):
        tr_rows, tr_sum = _track_record(tuple(sorted(symbols)), interval,
                                        not live)
    if tr_rows:
        st.caption(f"{tr_sum['calls']} replayed calls \u00b7 "
                   f"{tr_sum['wins']}/{tr_sum['graded']} graded wins "
                   f"({tr_sum['win_rate']:.0%}) \u00b7 "
                   f"avg return {tr_sum['avg_return_pct']}%")
        st.dataframe(pd.DataFrame(tr_rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No signals fired at the replay checkpoints.")

    st.divider()
    st.download_button("Download report (markdown)", render_markdown(res),
                       file_name="alpha-council-report.md")
    st.caption(FULL)
elif run:
    st.error("Pick at least one coin.")
