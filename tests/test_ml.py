"""ML signal tests. Run: python tests/test_ml.py (or pytest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphacouncil.ml import build_features, fit, ml_signal


def make_pattern_df(n=400, seed=7):
    """Series with a real (learnable) pattern: returns tend to persist."""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = 0.6 * rets[i - 1] + rng.normal(0, 0.01)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": np.roll(close, 1), "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": rng.uniform(500, 1500, n),
    }, index=idx)


def test_probabilities_are_valid():
    df = make_pattern_df()
    feats = build_features(df).dropna()
    y = (df["close"].shift(-3) / df["close"] - 1 > 0).astype(float).loc[feats.index].to_numpy()
    model = fit(feats, y)
    p = model.predict_proba(feats)
    assert ((p >= 0) & (p <= 1)).all()


def test_learns_persistent_pattern_above_chance():
    df = make_pattern_df()
    feats = build_features(df)
    future = df["close"].shift(-3) / df["close"] - 1
    y_all = (future > 0.003).astype(float)
    split = int(len(df) * 0.7)
    tr_X = feats.iloc[:split].dropna()
    tr_y = y_all.iloc[:split].loc[tr_X.index].to_numpy()
    te_X = feats.iloc[split:].dropna()
    te_y = y_all.iloc[split:].loc[te_X.index].to_numpy()
    model = fit(tr_X, tr_y)
    pred = model.predict_proba(te_X) > 0.5
    acc = (pred == te_y).mean()
    assert acc > 0.55, f"accuracy {acc:.2f} not above chance on learnable data"


def test_signal_only_in_test_window():
    df = make_pattern_df()
    sig = ml_signal(df, horizon=3, threshold=0.5, fit_frac=0.7)
    split = int(len(df) * 0.7)
    assert sig.iloc[:split].sum() == 0.0  # no signals inside the fit window
    assert set(sig.unique()) <= {0.0, 1.0}


def test_degenerate_labels_give_flat_signal():
    idx = pd.date_range("2025-01-01", periods=300, freq="D", tz="UTC")
    close = pd.Series(np.full(300, 100.0), index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1.0})
    sig = ml_signal(df)
    assert sig.sum() == 0.0


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"pass: {name}")
    print("all ml tests passed")
