"""Market data: free public APIs, no key required, with CSV caching.

Sources (tried in order):
  1. local CSV cache (data/samples or a user cache dir) - offline mode
  2. Kraken public OHLC (keyless, works everywhere)
  3. Binance public klines (keyless, geo-restricted in some regions)

Every frame is normalized to columns: timestamp (UTC), open, high, low,
close, volume. Bundled sample CSVs (BTC/ETH/SOL, 1d and 1h, real Kraken
data) let the whole repo run offline.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

KRAKEN_PAIRS = {"BTCUSD": "XBTUSD", "ETHUSD": "ETHUSD", "SOLUSD": "SOLUSD",
                "XRPUSD": "XRPUSD", "DOGEUSD": "DOGEUSD", "ADAUSD": "ADAUSD"}
BINANCE_SYMBOLS = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT",
                   "XRPUSD": "XRPUSDT", "DOGEUSD": "DOGEUSDT", "ADAUSD": "ADAUSDT"}
INTERVALS = {"1h": {"kraken": 60, "binance": "1h"}, "1d": {"kraken": 1440, "binance": "1d"}}

_UA = {"User-Agent": "alpha-council/0.1 (research; no trading)"}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _normalize(ts, o, h, l, c, v) -> pd.DataFrame:
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }).astype({"open": float, "high": float, "low": float,
               "close": float, "volume": float})
    return df.set_index("timestamp")


def from_kraken(symbol: str, interval: str) -> pd.DataFrame:
    pair = KRAKEN_PAIRS[symbol]
    url = (f"https://api.kraken.com/0/public/OHLC?pair={pair}"
           f"&interval={INTERVALS[interval]['kraken']}")
    data = _get_json(url)
    if data.get("error"):
        raise RuntimeError(f"kraken error: {data['error']}")
    rows = list(data["result"].values())[0]
    return _normalize([int(r[0]) for r in rows], [r[1] for r in rows],
                      [r[2] for r in rows], [r[3] for r in rows],
                      [r[4] for r in rows], [r[6] for r in rows])


def from_binance(symbol: str, interval: str) -> pd.DataFrame:
    sym = BINANCE_SYMBOLS[symbol]
    url = (f"https://api.binance.com/api/v3/klines?symbol={sym}"
           f"&interval={INTERVALS[interval]['binance']}&limit=1000")
    rows = _get_json(url)
    if isinstance(rows, dict):
        raise RuntimeError(f"binance error: {rows}")
    return _normalize([r[0] / 1000 for r in rows], [r[1] for r in rows],
                      [r[2] for r in rows], [r[3] for r in rows],
                      [r[4] for r in rows], [r[5] for r in rows])


def from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return (df[["timestamp", "open", "high", "low", "close", "volume"]]
            .astype({"open": float, "high": float, "low": float,
                     "close": float, "volume": float})
            .set_index("timestamp"))


def load(symbol: str, interval: str = "1d", offline: bool = False,
         cache_dir: Path | None = None) -> pd.DataFrame:
    """Load OHLCV for symbol ('BTCUSD'), interval '1d' or '1h'.

    offline=True reads only the bundled/cached CSVs. Otherwise tries cache,
    then Kraken, then Binance, and writes successful fetches to the cache.
    """
    cache_dir = cache_dir or SAMPLES_DIR
    cached = cache_dir / f"{symbol}_{interval}.csv"
    if offline:
        if not cached.exists():
            raise FileNotFoundError(f"no offline data for {symbol} {interval} at {cached}")
        return from_csv(cached)
    if cached.exists():
        df = from_csv(cached)
        # cache is fresh enough if its last bar is within 2 intervals of now
        step = pd.Timedelta(hours=1) if interval == "1h" else pd.Timedelta(days=1)
        if df.index[-1] >= pd.Timestamp.utcnow() - 2 * step:
            return df
    last_err: Exception | None = None
    for source in (from_kraken, from_binance):
        try:
            df = source(symbol, interval)
            try:
                out = df.reset_index()
                out["timestamp"] = (out["timestamp"].view("int64") // 10**9).astype(int)
                cache_dir.mkdir(parents=True, exist_ok=True)
                out.to_csv(cached, index=False)
            except OSError:
                pass
            return df
        except Exception as exc:  # noqa: BLE001 - try the next source
            last_err = exc
            time.sleep(0.5)
    if cached.exists():
        return from_csv(cached)  # stale cache beats no data
    raise RuntimeError(f"all data sources failed for {symbol} {interval}: {last_err}")


def validate(df: pd.DataFrame, interval: str = "1d") -> list[str]:
    """Data-quality checks. Returns a list of problems (empty = clean).
    Bad market data should never silently reach the strategy engine."""
    issues: list[str] = []
    if df is None or df.empty:
        return ["empty frame"]
    if not df.index.is_monotonic_increasing:
        issues.append("timestamps not chronological")
    if df.index.has_duplicates:
        issues.append(f"{int(df.index.duplicated().sum())} duplicate timestamps")
    ohlc = df[["open", "high", "low", "close"]]
    bad = (df["low"] > ohlc.min(axis=1) + 1e-9) | (df["high"] < ohlc.max(axis=1) - 1e-9)
    if bad.any():
        issues.append(f"{int(bad.sum())} bars with inconsistent OHLC")
    if (df["volume"] < 0).any():
        issues.append("negative volume")
    if ohlc.isna().any().any():
        issues.append("NaNs in OHLC")
    if (ohlc <= 0).any().any():
        issues.append("non-positive prices")
    freq = pd.tseries.frequencies.to_offset("1D" if interval == "1d" else "1h")
    gaps = df.index.to_series().diff().dropna()
    missing = gaps[gaps > freq * 1.5]
    if len(missing):
        issues.append(f"{len(missing)} missing-bar gaps")
    jumps = df["close"].pct_change().abs()
    if (jumps > 0.5).any():
        issues.append(f"{int((jumps > 0.5).sum())} extreme price jumps (>50%)")
    return issues
