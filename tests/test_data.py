import numpy as np
import pandas as pd

from alphacouncil.data import validate


def _clean(n=50):
    idx = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.full(n, 10.0)}, index=idx)


def test_clean_frame_passes():
    assert validate(_clean()) == []


def test_duplicate_timestamp_detected():
    df = _clean()
    df = pd.concat([df, df.iloc[[5]]]).sort_index()
    assert any("duplicate" in i for i in validate(df))


def test_bad_ohlc_detected():
    df = _clean()
    df.loc[df.index[10], "low"] = df["close"].iloc[10] + 5
    assert any("OHLC" in i for i in validate(df))


def test_gap_detected():
    df = _clean().drop(_clean().index[10:14])
    assert any("gap" in i for i in validate(df))


def test_nan_detected():
    df = _clean()
    df.loc[df.index[3], "close"] = np.nan
    assert any("NaN" in i for i in validate(df))


def test_extreme_jump_detected():
    df = _clean()
    df.loc[df.index[20]:, "close"] *= 3
    assert any("jump" in i for i in validate(df))
