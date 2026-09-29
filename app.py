"""alpha-council dashboard (Streamlit).

    pip install -r requirements.txt
    streamlit run app.py

Runs offline out of the box (bundled data + mock council). Add a free-tier
API key in the sidebar for a live LLM council on fresh market data.

Suggestions only - not financial advice.
"""

from __future__ import annotations

import streamlit as st

from alphacouncil.data import KRAKEN_PAIRS, load
from alphacouncil.desk import run_desk
from alphacouncil.disclaimers import FULL, SHORT
from alphacouncil.report import render_markdown

st.set_page_config(page_title="alpha-council", page_icon="📊", layout="wide")

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
            st.error("Live council needs an API key (free tier is fine).")
            st.stop()
        base_url, default_model, _ = FREE_PROVIDERS[provider]
        model = OpenAIModel(default_model, base_url=base_url, api_key=api_key)

    with st.spinner("Desk is working: data -> strategies -> walk-forward "
                    "backtests -> council debate..."):
        res = run_desk(symbols, interval, offline=not live, model=model)

    st.subheader("Market view")
    st.write(res.market_view)

    approved = [c for c in res.candidates if not c.vetoed]
    vetoed = [c for c in res.candidates if c.vetoed]

    st.subheader(f"Trade suggestions ({len(approved)})")
    if not approved:
        st.info("No live setups passed risk policy. No suggestion IS the suggestion.")
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

    st.divider()
    st.download_button("Download report (markdown)", render_markdown(res),
                       file_name="alpha-council-report.md")
    st.caption(FULL)
elif run:
    st.error("Pick at least one coin.")
