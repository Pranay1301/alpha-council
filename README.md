# alpha-council

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
  (train on 70%, score on                     (code enforces the same
  unseen 30%, fees + slippage)                rules - model veto is advisory)
        │                                              │
        └──────────►  suggestion cards  ◄────  critic: writes the bear case
                     entry / stop / target             for every approval
                     + out-of-sample stats
```

The separation is the point: **models write prose, code owns math.** An
LLM cannot move a stop-loss, invent a win rate, or talk a losing strategy
onto the list - the risk policy in code rejects it regardless of what any
agent says.

## What a suggestion looks like (real offline run)

```
### BTCUSD - LONG - trend_following
- entry ~ 84,288 | stop 79,815 | target 93,235 (R:R 2.0, ATR-based stop)
- backtest (out-of-sample): 13 trades, win rate 61.5%, profit factor 1.26,
  max drawdown 15.2%, Sharpe 0.44
- thesis (strategist agent): ...
- bear case (critic agent): ...
```

Every card carries its **out-of-sample** backtest stats. When nothing
passes the risk policy, the desk says so - no suggestion IS the suggestion.

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
- Parameter selection is **walk-forward**: grid-search on the first 70%
  of history, then scored once on the held-out 30%. The number you see on
  a card is the out-of-sample score, never the training score.

Known limits (read these before trusting any backtest, here or anywhere):
no liquidity/slippage modeling beyond a flat assumption, no funding or
borrow costs, regime changes break edges, and 2 years of daily bars is a
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

- [ ] paper-trading ledger: log suggestions, track outcomes vs stop/target
- [ ] walk-forward with multiple rolling windows (not just one 70/30 cut)
- [ ] short-side setups (perpetuals data) with funding-rate awareness
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
