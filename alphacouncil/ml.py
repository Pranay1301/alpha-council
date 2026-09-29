"""A simple ML signal: logistic regression written in numpy, trained on
historical bars to predict whether the next `horizon`-bar return beats the
round-trip cost (fees + slippage).

Honesty rules baked in:
- The model trains ONLY on the first `fit_frac` of whatever frame it is
  given and emits signals for the rest. No future data leaks into fit:
  features are standardized with train-window statistics only.
- It is one signal among several, not an oracle. Backtest metrics decide
  whether it earns a place in a suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr, ema, macd, rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ret1 = close.pct_change()
    macd_line, _, hist = macd(close)
    feats = pd.DataFrame({
        "ret_1": ret1,
        "ret_3": close.pct_change(3),
        "ret_5": close.pct_change(5),
        "rsi": rsi(close) / 100.0,
        "macd_hist": hist / close,
        "atr_pct": atr(df["high"], df["low"], close) / close,
        "ema20_dist": (close - ema(close, 20)) / close,
        "vol_z": ((df["volume"] - df["volume"].rolling(20).mean())
                  / df["volume"].rolling(20).std(ddof=0)),
    })
    return feats.replace([np.inf, -np.inf], np.nan)


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.feature_names].to_numpy() - self.mean) / self.std
        Z = np.nan_to_num(Z)
        logits = Z @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))


def fit(X: pd.DataFrame, y: np.ndarray, *, lr: float = 0.1, epochs: int = 400,
        l2: float = 1e-3) -> LogisticModel:
    mean = X.mean().to_numpy()
    std = X.std(ddof=0).replace(0, 1).to_numpy()
    Z = np.nan_to_num((X.to_numpy() - mean) / std)
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(Z)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        grad_w = Z.T @ (p - y) / n + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return LogisticModel(w, b, mean, std, list(X.columns))


def ml_signal(df: pd.DataFrame, *, horizon: int = 5, threshold: float = 0.55,
              fit_frac: float = 0.7, cost_bps: float = 30.0) -> pd.Series:
    """Long/flat series. Fit on the first fit_frac of df, signal the rest."""
    feats = build_features(df)
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = (future_ret > cost_bps / 1e4).astype(float)

    split = int(len(df) * fit_frac)
    train_X = feats.iloc[:split].dropna()
    train_y = y.iloc[:split].loc[train_X.index].to_numpy()
    if train_X.empty or train_y.sum() in (0, len(train_y)):
        return pd.Series(0.0, index=df.index)

    model = fit(train_X, train_y)
    test_mask = pd.Series(feats.index >= feats.index[split], index=feats.index)
    proba = pd.Series(model.predict_proba(feats.dropna()),
                      index=feats.dropna().index)
    sig = ((proba > threshold) & test_mask.reindex(proba.index).fillna(False)).astype(float)
    return sig.reindex(df.index).fillna(0.0)
