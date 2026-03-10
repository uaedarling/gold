"""
signal_engine/entry_detector.py — M5 precision entry detection for GOLD_QUANT_AI.

Confirms a high-probability entry point on the M5 chart by requiring three
aligned smart-money signals: a liquidity sweep that traps weak hands, a Break
of Structure that confirms the new directional move, and a momentum candle that
proves participation.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_entry_m5(m5: pd.DataFrame, direction: str) -> bool:
    """Detect a valid entry setup on the M5 chart.

    Entry criteria (all three must be true)
    ----------------------------------------
    1. **Liquidity sweep** in the direction of the trade:
       - BUY → a ``liquidity_sweep_low`` occurred in the last 5 bars.
       - SELL → a ``liquidity_sweep_high`` occurred in the last 5 bars.
    2. **Break of Structure** in the trade direction:
       - BUY → ``bos_up`` is True on the most recent bar.
       - SELL → ``bos_down`` is True on the most recent bar.
    3. **Momentum** — ``candle_range`` of the last bar exceeds ``atr_14``,
       indicating a strong directional move.

    Parameters
    ----------
    m5:
        Feature-enriched M5 DataFrame containing ``liquidity_sweep_low``,
        ``liquidity_sweep_high``, ``bos_up``, ``bos_down``, ``candle_range``,
        and ``atr_14`` columns.
    direction:
        Trade direction — ``"BUY"`` or ``"SELL"``.

    Returns
    -------
    bool
        ``True`` when all entry criteria are satisfied.
    """
    if m5 is None or m5.empty:
        return False

    required = {
        "liquidity_sweep_low",
        "liquidity_sweep_high",
        "bos_up",
        "bos_down",
        "candle_range",
        "atr_14",
    }
    if not required.issubset(m5.columns):
        logger.warning("M5 missing columns for entry detection: %s", required - set(m5.columns))
        return False

    recent = m5.dropna(subset=["candle_range", "atr_14"]).tail(5)
    if recent.empty:
        return False

    last = recent.iloc[-1]

    # Criterion 1: liquidity sweep
    if direction == "BUY":
        liq_ok = bool(recent["liquidity_sweep_low"].any())
    elif direction == "SELL":
        liq_ok = bool(recent["liquidity_sweep_high"].any())
    else:
        liq_ok = False

    # Criterion 2: break of structure
    if direction == "BUY":
        bos_ok = bool(last["bos_up"])
    elif direction == "SELL":
        bos_ok = bool(last["bos_down"])
    else:
        bos_ok = False

    # Criterion 3: momentum candle
    momentum_ok = last["candle_range"] > last["atr_14"]

    logger.debug(
        "Entry check: direction=%s, liq_ok=%s, bos_ok=%s, momentum_ok=%s",
        direction,
        liq_ok,
        bos_ok,
        momentum_ok,
    )
    return liq_ok and bos_ok and momentum_ok
