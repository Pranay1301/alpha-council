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

from alphacouncil.data import KRAKEN_PAIRS, load, probe
from alphacouncil.indicators import ema
from alphacouncil.desk import BT_KW
from alphacouncil.desk import run_desk
from alphacouncil.disclaimers import FULL, SHORT
from alphacouncil.report import render_markdown

st.set_page_config(page_title="alpha-council", page_icon="📊", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def _track_record(symbols: tuple, interval: str, offline: bool):
    from alphacouncil.trackrecord import replay, summary
    rows = replay(list(symbols), interval, offline)
    return rows, summary(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def _offline_desk(symbols: tuple[str, ...], interval: str):
    # Cache deterministic offline quant; never cache a live LLM result or API key.
    return run_desk(list(symbols), interval, offline=True, model=None)


def _candles(df: pd.DataFrame, title: str, candidate=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = df.tail(120)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="OHLC"), row=1, col=1)
    for n, color in ((20, "#46b3ff"), (50, "#ffbd59")):
        fig.add_trace(go.Scatter(x=df.index, y=ema(df["close"], n),
                                 mode="lines", name=f"EMA{n}",
                                 line=dict(color=color, width=1.5)), row=1, col=1)
    if candidate is not None:
        for key, color in (("entry", "#45cf89"), ("stop", "#fa646e"),
                           ("target", "#e8ba4d")):
            fig.add_hline(y=getattr(candidate, key), line_dash="dash",
                          line_color=color, annotation_text=key.title(), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color="#64748b"), row=2, col=1)
    fig.update_layout(height=430, title=title, template="plotly_dark",
                      xaxis_rangeslider_visible=False, margin=dict(t=50, b=10, l=5, r=10))
    return fig

st.title("alpha-council")
st.caption("An AI research desk for crypto. LLM agents propose, rank and "
           "attack trade setups; code owns every number; risk policy can "
           "veto anything.")
st.warning(SHORT)
st.markdown("**Research workflow**  Market data → rolling out-of-sample tests → "
            "council debate → hard risk veto. No setup is a valid result.")
st.caption("Markets: BTC · ETH · SOL | Modes: bundled offline sample or fresh live "
           "data | Council: analyst, strategist, risk manager, critic")

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
    try:
        from alphacouncil.model import FREE_PROVIDERS
        configured_key = st.secrets.get(FREE_PROVIDERS[provider][2], "") if live else ""
    except Exception:
        configured_key = ""
    api_key = ""
    if live:
        if configured_key:
            st.success("Council connected via deployment secret. No key to paste.")
        else:
            api_key = st.text_input("API key", type="password",
                                    help=f"Free key from the {provider} console. "
                                         "Not stored by the app.")
    run = st.button("Run the desk", type="primary", width="stretch")

if run and symbols:
    model = None
    if live:
        import os
        from alphacouncil.model import FREE_PROVIDERS, OpenAIModel
        if not api_key:
            api_key = configured_key
        if not api_key:
            st.error("Live council needs an API key (free tier is fine).")
            st.stop()
        base_url, default_model, _ = FREE_PROVIDERS[provider]
        model = OpenAIModel(default_model, base_url=base_url, api_key=api_key)

    with st.spinner("Desk is working: data -> strategies -> walk-forward "
                    "backtests -> council debate..."):
        try:
            res = (run_desk(symbols, interval, offline=False, model=model)
                   if live else _offline_desk(tuple(symbols), interval))
        except Exception:  # noqa: BLE001 - public app must not disclose secrets
            import logging
            logging.getLogger(__name__).exception("Desk run failed")
            st.error("The desk run could not finish. Please try again later. "
                     "No trade suggestion was issued.")
            st.stop()

    dstat = []
    stale = False
    for s in res.symbols:
        p = probe(s, interval, not live)
        if "error" not in p:
            dstat.append(f"**{s}**: {p['source']}, last bar {p['last_bar']}")
            stale = stale or not p["fresh"]
    if dstat:
        st.caption("Data: " + " \u00b7 ".join(dstat))
    if stale:
        st.warning("Data is stale - results should not be interpreted as current.")

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
    # Streamlit can rerun new app.py while an older desk module stays imported.
    # Calculate a display fallback from the same data until a full restart.
    betas = getattr(res, "beta_to_btc", {})
    if not betas and "BTCUSD" in res.symbols:
        try:
            btc = load("BTCUSD", interval, offline=not live)["close"].pct_change(fill_method=None).tail(180)
            for sym in res.symbols:
                ret = load(sym, interval, offline=not live)["close"].pct_change(fill_method=None).tail(180)
                aligned = pd.concat([ret, btc], axis=1, join="inner").dropna()
                variance = float(aligned.iloc[:, 1].var()) if len(aligned) >= 30 else 0.0
                if variance > 1e-12:
                    betas[sym] = round(float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance), 2)
        except Exception:
            pass  # beta unavailable is not a reason to suppress the desk
    ov_rows = []
    for s in res.symbols:
        dd = res.regime.get(s, {}) if isinstance(res.regime, dict) else {}
        n_ok = len([c for c in res.candidates if c.symbol == s and not c.vetoed])
        ov_rows.append({"Coin": s, "Regime": dd.get("regime", "-"),
                        "Trend (EMA20/50)": dd.get("ema20_vs_ema50", "-"),
                        "ATR %": dd.get("atr_pct", "-"),
                        "BTC beta (180 bars)": betas.get(s, "-"),
                        "Approved setups": n_ok})
    st.dataframe(pd.DataFrame(ov_rows), width="stretch", hide_index=True)
    st.caption("BTC beta is historical covariance / BTC variance on aligned "
               "close-to-close returns (up to 180 bars), not a forecast. "
               "Theme concentration is not modeled.")
    with st.expander("Market charts (120 bars · candles, EMA20/50, volume)"):
        for sym in res.symbols:
            try:
                st.plotly_chart(_candles(load(sym, interval, offline=not live), sym),
                                width="stretch")
            except Exception as exc:
                st.caption(f"{sym} chart unavailable: {exc}")

    approved = [c for c in res.candidates if not c.vetoed]
    vetoed = [c for c in res.candidates if c.vetoed]

    st.subheader(f"Trade suggestions ({len(approved)})")
    if approved:
        st.caption(f"{len(approved)} setups found \u00b7 "
                   f"{res.distinct_opportunities} materially distinct "
                   f"opportunities"
                   + (f" \u00b7 correlations: "
                      + ", ".join(f"{k} {v}" for k, v in res.correlation.items())
                      if res.correlation else ""))
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
                if c.evidence:
                    ev = c.evidence
                    comp = " \u00b7 ".join(f"{k} {v:.2f}" for k, v in ev["components"].items())
                    st.markdown(f"**Evidence score: {ev['total']}/100** (deterministic, not LLM)  \u2003{comp}")
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
                st.plotly_chart(_candles(df, c.symbol, c), width="stretch")
            except Exception:
                pass

    with st.expander("Backtest methodology and limits"):
        st.markdown(f"**Evaluation:** 3 expanding training windows, each followed "
                    f"by a held-out test. ML is refit on each training window. "
                    f"Reported profit factor is the median across folds; drawdown "
                    f"is the worst fold. Train parameters are frozen on each test.\n\n"
                    f"**Execution:** signal at bar close, next-bar open entry, "
                    f"stop/target checks from subsequent bars. A bar touching both "
                    f"counts as a stop. Stop distance {BT_KW['stop_atr']}× ATR, "
                    f"target {BT_KW['rr']}× risk, fees {BT_KW['fee_bps']:.0f} bps "
                    f"each side, slippage {BT_KW['slippage_bps']:.0f} bps on "
                    f"entry/exit, maximum hold 30 bars.\n\n"
                    "**Limits:** sample size, regime changes, flat transaction costs "
                    "and no funding or liquidity model. Historical results are not "
                    "a forecast. The replay below is not a live paper-trading ledger.")
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
        st.dataframe(lb, width="stretch", hide_index=True)

    if res.council_log:
        with st.expander("Full council debate"):
            for stage in res.council_log:
                st.markdown(f"**{stage['stage']}**")
                st.text(stage["output"][:2000])

    if res.audit_log:
        with st.expander("Council run audit metadata"):
            st.caption("Hashes identify inputs, prompts and outputs without "
                       "publishing their contents. No API key is recorded.")
            st.dataframe(pd.DataFrame(res.audit_log), width="stretch",
                         hide_index=True)

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
                   f"avg return {tr_sum['avg_return_pct']}% · "
                   f"profit factor "
                   f"{tr_sum['profit_factor'] if tr_sum['profit_factor'] is not None else 'undefined (no losses)'}")
        st.dataframe(pd.DataFrame(tr_rows), width="stretch",
                     hide_index=True)
        st.caption("MAE/MFE bound the move through the exit bar; intrabar order "
                   "is unknown. This is retrospective replay, not a persistent "
                   "paper-trading ledger or observed live performance.")
    else:
        st.info("No signals fired at the replay checkpoints.")

    st.divider()
    st.download_button("Download report (markdown)", render_markdown(res),
                       file_name="alpha-council-report.md")
    st.caption(FULL)
elif run:
    st.error("Pick at least one coin.")
