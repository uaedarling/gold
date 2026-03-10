"""
signal_engine/pullback_detector.py — M15 pullback detection for GOLD_QUANT_AI.

Identifies when price has pulled back to a key level (EMA-50) in the context
of an established trend, confirmed by RSI being in the expected retracement
zone.
"""

import logging

import pandas as pd

from gold_quant_ai.config import EMA_PROXIMITY_PCT

logger = logging.getLogger(__name__)

# RSI bands that define a "healthy" pullback in each direction
_BULLISH_RSI_LOW = 35
_BULLISH_RSI_HIGH = 55
_BEARISH_RSI_LOW = 45
_BEARISH_RSI_HIGH = 65


def detect_pullback_m15(m15: pd.DataFrame, trend: str) -> bool:
    """Detect whether a valid pullback to the EMA-50 is occurring on M15.

    Criteria
    --------
    * Current close is within :data:`_EMA_PROXIMITY_PCT`% of the EMA-50.
    * For a **BULLISH** trend: RSI-14 is between 35 and 55 (oversold retest).
    * For a **BEARISH** trend: RSI-14 is between 45 and 65 (overbought retest).

    Parameters
    ----------
    m15:
        Feature-enriched M15 DataFrame containing ``close``, ``ema_50``, and
        ``rsi_14`` columns.
    trend:
        The current directional bias — ``"BULLISH"`` or ``"BEARISH"``.

    Returns
    -------
    bool
        ``True`` when all pullback criteria are satisfied.
    """
    if m15 is None or m15.empty:
        return False

    required = {"close", "ema_50", "rsi_14"}
    if not required.issubset(m15.columns):
        logger.warning("M15 missing columns for pullback detection: %s", required - set(m15.columns))
        return False

    latest = m15.dropna(subset=list(required))
    if latest.empty:
        return False

    last = latest.iloc[-1]
    close = last["close"]
    ema50 = last["ema_50"]
    rsi = last["rsi_14"]

    if ema50 == 0:
        return False

    # Price proximity check
    pct_from_ema = abs(close - ema50) / ema50 * 100
    if pct_from_ema > EMA_PROXIMITY_PCT:
        logger.debug("Pullback check: pct_from_ema=%.3f%% > threshold → no pullback.", pct_from_ema)
        return False

    # RSI zone check
    if trend == "BULLISH":
        rsi_ok = _BULLISH_RSI_LOW <= rsi <= _BULLISH_RSI_HIGH
    elif trend == "BEARISH":
        rsi_ok = _BEARISH_RSI_LOW <= rsi <= _BEARISH_RSI_HIGH
    else:
        rsi_ok = False

    logger.debug(
        "Pullback check: trend=%s, pct_from_ema=%.3f%%, rsi=%.1f, rsi_ok=%s",
        trend,
        pct_from_ema,
        rsi,
        rsi_ok,
    )
    return rsi_ok
