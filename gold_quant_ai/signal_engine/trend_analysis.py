"""
signal_engine/trend_analysis.py — Multi-timeframe trend / bias analysis.

Determines the macro market bias from the H4 timeframe using EMA-200 and
confirms intermediate trend structure on H1 using swing-high/swing-low
pattern recognition.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def market_bias_h4(h4: pd.DataFrame) -> str:
    """Determine the macro directional bias from the H4 chart.

    The bias is considered **BULLISH** when the most recent close is above the
    200-period EMA, and **BEARISH** otherwise.

    Parameters
    ----------
    h4:
        Feature-enriched H4 DataFrame that must contain ``close`` and
        ``ema_200`` columns.

    Returns
    -------
    str
        ``"BULLISH"`` or ``"BEARISH"``.
    """
    if h4 is None or h4.empty or "ema_200" not in h4.columns:
        logger.warning("H4 data or ema_200 missing — defaulting bias to BEARISH.")
        return "BEARISH"

    latest = h4.dropna(subset=["close", "ema_200"])
    if latest.empty:
        return "BEARISH"

    last = latest.iloc[-1]
    bias = "BULLISH" if last["close"] > last["ema_200"] else "BEARISH"
    logger.debug("H4 bias: close=%.2f, ema_200=%.2f → %s", last["close"], last["ema_200"], bias)
    return bias


def confirm_trend_h1(h1: pd.DataFrame) -> str:
    """Confirm intermediate trend direction from the H1 chart.

    Uses the last 20 rows of swing structure columns (``hh``, ``hl``, ``lh``,
    ``ll``) to determine whether price is making Higher Highs + Higher Lows
    (BULLISH), Lower Highs + Lower Lows (BEARISH), or neither (NEUTRAL).

    Parameters
    ----------
    h1:
        Feature-enriched H1 DataFrame that must contain boolean columns
        ``hh``, ``hl``, ``lh``, and ``ll``.

    Returns
    -------
    str
        ``"BULLISH"``, ``"BEARISH"``, or ``"NEUTRAL"``.
    """
    if h1 is None or h1.empty:
        return "NEUTRAL"

    required = {"hh", "hl", "lh", "ll"}
    if not required.issubset(h1.columns):
        logger.warning("H1 structure columns missing: %s", required - set(h1.columns))
        return "NEUTRAL"

    recent = h1.tail(20)
    bullish_signals = int(recent["hh"].any()) + int(recent["hl"].any())
    bearish_signals = int(recent["lh"].any()) + int(recent["ll"].any())

    if bullish_signals > bearish_signals:
        trend = "BULLISH"
    elif bearish_signals > bullish_signals:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    logger.debug("H1 trend: bullish_signals=%d, bearish_signals=%d → %s", bullish_signals, bearish_signals, trend)
    return trend
