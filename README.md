# alpha-council

**[Live dashboard](https://alpha-council-6hkkwmjbnsuyqd8rfnc494.streamlit.app)**

**An AI research desk for crypto.** LLM agents propose, rank, veto and
attack trade setups - but every number (entry, stop-loss, take-profit,
backtest stats) is computed by code, never by a model. Setups are
backtested out-of-sample before they can appear. Suggestions only.

> **Research tool. Suggestions only - not financial advice.** Backtests
> describe the past, not the future. See the full disclaimer at the bottom.

```
python examples/run_desk.py            # offline: bundled real data + mock council, no API key
GROQ_API_KEY=... python examples/run_desk.py --live --provider groq   # live council, free tier
streamlit run app.py                   # the dashboard
```

## How it works

```
market data (free public APIs)                LLM council (free-tier models)
        │                                              │
        ▼                                              ▼
  indicators (RSI, MACD,          analyst:    reads the quant summary,
  Bollinger, ATR - from scratch)              writes the market view
        │                                              │
  strategies (trend, mean-        strategist: ranks live setups, writes
  reversion, breakout, ML)                    the thesis for each
        │                                              │
  walk-forward backtester         risk mgr:   vetoes weak evidence
  (3 expanding train/test                   (code enforces the same
  folds, fees + slippage)                    rules - model veto is advisory)
        │                                              │
        └──────────►  suggestion cards  ◄────  critic: writes the bear case
                     entry / stop / target             for every approval
                     + out-of-sample stats
```

The separation is the point: **models write prose, code owns math.** An
LLM cannot move a stop-loss, invent a win rate, or talk a losing strategy
onto the list - the risk policy in code rejects it regardless of what any
agent says.

## Reading the dashboard

Use **Run the desk** in quant-only mode for a reproducible offline sample.
Turn on the live council for fresh data and LLM analysis. A deployment
secret connects Groq without showing an API-key field; other providers
can be configured separately. The Market overview, candlestick/EMA/volume
charts, leaderboard, backtest methodology and track-record replay stay
available when no setup passes the policy.

**The current bundled daily sample has zero approved suggestions.** BTC
trend-following has median held-out profit factor 0.95; SOL trend-following
has 0.58, both below the 1.0 veto threshold. The old single-split scores
should not be used as evidence. The replay table reruns the desk's **exact decision policy** (candidate
generation, hard risk vetoes, evidence, portfolio layer - no hindsight
strategy selection) at past checkpoints on truncated historical data, and
grades each call with the **same canonical execution** as the backtester:
entry at the next bar's open plus slippage, stop/target from the following
bar, stop wins collisions, fees both sides. It shows bars held and MAE/MFE;
it is retrospective replay, **not** observed live performance. The portfolio layer reports return correlation and historical BTC beta
(covariance/variance on up to 180 aligned return bars), and groups strongly
correlated approvals as one distinct opportunity; this is not a complete
portfolio sizing or thematic exposure model.

Each live council stage records provider, model, UTC timestamp, SHA-256
prompt/input/output hashes, latency and token usage in the exported report.
No API key or raw model input is included in that metadata.

One canonical `simulate_trade`/`monitor_trade` pair
(`alphacouncil/backtest.py`) is shared by the backtester, the historical
replay, the paper ledger and the effectiveness study, so these views can
never silently apply different rules. Walk-forward results report each
fold's own parameters and metrics separately from the current live
(modal-across-folds) parameters. Council outputs are strictly
schema-validated against the live candidate set: unknown keys are dropped,
indices must be unique and in range, prose is length-capped.

The dashboard also carries:

- a **paper ledger** (`alphacouncil/paper.py`): approved setups are
  snapshotted at signal time and resolved bar-by-bar
  (NEW -> ACTIVE -> TARGET/STOP/EXPIRED) against new data. Stored as JSON
  in the app filesystem - durable when self-hosted, ephemeral on free
  Streamlit hosting (the app says so and offers a download). Paper trades,
  not real ones;
- a **council effectiveness** paired replay
  (`alphacouncil/effectiveness.py`): quant-only vs quant+council on
  identical historical opportunities, with veto precision/recall when a
  live model is attached. Samples are small and descriptive, not proof;
- a **research integrity** panel listing the guarantees and the known
  limits (no liquidity model, no funding costs, limited sample,
  correlated assets).

## The strategies

| strategy | idea | parameters chosen by |
|---|---|---|
| `trend_following` | EMA fast/slow cross with an RSI cap | walk-forward grid search |
| `mean_reversion` | RSI oversold + below lower Bollinger band | walk-forward grid search |
| `breakout` | close above prior N-bar high on volume surge | walk-forward grid search |
| `ml_logistic` | logistic regression (from scratch, numpy) predicts whether the next 5 bars beat round-trip cost | fit on train window only |

## The backtester, honestly

Execution rules in `alphacouncil/backtest.py`:

- Signals compute at bar close; entries fill at the **next** bar's open.
  No lookahead.
- Fees on both sides (10 bps default) and slippage (5 bps) on entry/exit.
- If one bar touches both stop and target, the **stop is assumed hit
  first**. Conservative on purpose.
- Exits: stop-loss, take-profit, signal-off, or max holding period.
- Parameter selection is **rolling walk-forward**: three expanding training
  folds, each followed by an unseen test segment. ML is refit on each
  training fold. Cards show median test profit factor, worst test drawdown,
  total test trades, profitable-fold share and parameter stability. The
  method does not guarantee independence across correlated regimes.

Known limits (read these before trusting any backtest, here or anywhere):
no liquidity/slippage modeling beyond a flat assumption, no funding or
borrow costs, regime changes break edges, and limited daily history is a
small sample. A good out-of-sample result is evidence, not a promise.

## Risk policy (enforced in code, not by prompt)

A setup cannot be suggested if, out-of-sample: profit factor < 1.0, or
fewer than 4 trades, or max drawdown > 35%. The risk-agent's LLM veto is
an extra, advisory layer on top.

## Free-tier LLMs

The council runs on free, OpenAI-compatible providers - no paid key needed:

| provider | model | get a key |
|---|---|---|
| Groq (default) | `openai/gpt-oss-120b` | console.groq.com |
| z.ai | `glm-4.5-flash` ($0) | z.ai |
| OpenRouter | `openrouter/free` | openrouter.ai |

```
ZAI_API_KEY=...        python examples/run_desk.py --live --provider zai
OPENROUTER_API_KEY=... python examples/run_desk.py --live --provider openrouter
```

## Market data

Free, keyless public APIs: Kraken (primary, works everywhere) and Binance
(fallback, geo-restricted in some regions). Everything fetched is cached
to CSV. The repo bundles real sample data (BTC/ETH/SOL, ~2 years daily +
30 days hourly, fetched from Kraken on 2026-09-29) so the whole thing -
demo, tests, dashboard - runs offline.

## Tests

```
python tests/test_indicators.py   # RSI/ATR match Wilder's definitions, hand-checked
python tests/test_backtest.py     # no-lookahead entries, stop-first rule, fee math
python tests/test_ml.py           # learns a real pattern, no leakage into fit window
python tests/test_desk.py         # end-to-end offline run, risk policy enforced
```

## Roadmap

- [x] paper-trading ledger: suggestions snapshotted at signal time, resolved
      vs stop/target (JSON storage; durable when self-hosted)
- [x] walk-forward with multiple rolling windows (not just one 70/30 cut)
- [x] paired quant-only vs quant+council effectiveness replay
- [ ] durable hosted storage for the paper ledger (free DB)
- [ ] short-side setups (perpetuals data) with funding-rate awareness
- [ ] liquidity-aware execution, slippage sensitivity, bootstrap confidence
      intervals, Monte Carlo trade-sequence analysis
- [ ] scheduled runs with alerts (Telegram/WhatsApp) when a new setup appears

## Disclaimer

alpha-council is a research and education project. Nothing it produces is
financial advice, a recommendation to buy or sell, or a guarantee of
profit. Backtests describe the past, not the future: they can overfit,
ignore real-world liquidity, and break down when market regimes change.
Crypto assets are highly volatile and you can lose all of your money. Do
your own research and consider talking to a licensed financial advisor
before risking anything.

## License

MIT
