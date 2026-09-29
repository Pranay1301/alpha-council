"""Run the alpha-council research desk.

Offline (default): bundled real market data + scripted mock council. No API
key, no network, fully reproducible:

    python examples/run_desk.py

Live council with a free-tier LLM (real market data via Kraken/Binance):

    GROQ_API_KEY=... python examples/run_desk.py --live --provider groq
    ZAI_API_KEY=...  python examples/run_desk.py --live --provider zai --symbols BTCUSD,SOLUSD

Suggestions only - not financial advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphacouncil.desk import run_desk
from alphacouncil.model import MockModel, OpenAIModel
from alphacouncil.report import render_markdown


def mock_council() -> MockModel:
    """Scripted council for offline runs: analyst, strategist, risk, critic."""
    return MockModel([
        # analyst (summary payload is data-dependent, so keep prose generic)
        json.dumps({"market_view":
            "Across the covered majors the desk sees a mixed regime: RSI in "
            "neutral-to-strong bands and price stretched relative to the "
            "20/50-bar trend on some names, flat on others. Volatility "
            "(ATR%) is within its recent range. The desk is watching whether "
            "trend signals hold above their EMA50 anchors on the next bars."}),
        # strategist
        json.dumps({"ranked": "all",
                    "thesis": {}}),
        # risk
        json.dumps({"vetoes": {},
                    "notes": "Treat every suggestion as one position of many; "
                             "size so a full stop-out costs a small fixed "
                             "fraction of the account."}),
        # critic
        json.dumps({"bear_case": {}}),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--interval", choices=["1d", "1h"], default="1d")
    parser.add_argument("--live", action="store_true",
                        help="fetch fresh data and use a real LLM council")
    parser.add_argument("--provider", choices=["groq", "zai", "openrouter"],
                        default="groq")
    parser.add_argument("--model", default=None, help="override provider model")
    parser.add_argument("--out", default=None, help="write report markdown to this path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    model = (OpenAIModel.from_provider(args.provider, args.model)
             if args.live else mock_council())
    res = run_desk(symbols, args.interval, offline=not args.live, model=model)
    report = render_markdown(res)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\n[wrote {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
