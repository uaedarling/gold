"""
feature_engine/indicator_features.py — Technical indicator feature builder.

Adds a comprehensive set of technical indicators to an OHLCV DataFrame so
that downstream machine-learning models and signal filters have rich input
features without needing direct access to the raw TA library.
"""

import logging

import numpy as np
import pandas as pd

try:
    import ta  # type: ignore
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback pure-pandas implementations (used when *ta* is not installed)
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average using pandas ewm."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram.

    Returns
    -------
    tuple of pd.Series
        ``(macd, macd_signal, macd_hist)``
    """
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to *df* in-place and return it.

    The following columns are appended:

    * ``ema_50``              — 50-period Exponential Moving Average of close
    * ``ema_200``             — 200-period EMA of close
    * ``rsi_14``              — 14-period Relative Strength Index
    * ``atr_14``              — 14-period Average True Range
    * ``macd``                — MACD line (12/26 EMA difference)
    * ``macd_signal``         — 9-period EMA of the MACD line
    * ``macd_hist``           — MACD histogram (macd − macd_signal)
    * ``candle_range``        — high − low for each candle
    * ``avg_candle_range_20`` — 20-period rolling mean of candle_range
    * ``atr_expansion``       — atr_14 / (20-period rolling mean of atr_14)

    Parameters
    ----------
    df:
        DataFrame with at minimum the columns ``open``, ``high``, ``low``,
        ``close``, and ``volume``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with additional indicator columns appended.
    """
    df = df.copy()

    if _TA_AVAILABLE:
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        macd_obj = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()
    else:
        logger.warning("'ta' library not found — using pure-pandas fallback indicators.")
        df["ema_50"] = _ema(df["close"], 50)
        df["ema_200"] = _ema(df["close"], 200)
        df["rsi_14"] = _rsi(df["close"], 14)
        df["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])

    df["candle_range"] = df["high"] - df["low"]
    df["avg_candle_range_20"] = df["candle_range"].rolling(20).mean()

    rolling_atr_mean = df["atr_14"].rolling(20).mean()
    df["atr_expansion"] = df["atr_14"] / rolling_atr_mean.replace(0, np.nan)

    return df
