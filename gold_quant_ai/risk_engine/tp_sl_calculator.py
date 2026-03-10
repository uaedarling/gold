"""
risk_engine/tp_sl_calculator.py — Stop-loss and take-profit calculator for GOLD_QUANT_AI.

Uses recent swing levels identified in the M5 feature DataFrame to place
structurally sound stop-loss levels, then derives take-profit targets from a
configurable risk-to-reward ratio.
"""

import logging
from typing import Dict

import pandas as pd

from gold_quant_ai.config import DEFAULT_RR, ACCOUNT_BALANCE, RISK_PER_TRADE_PCT
from gold_quant_ai.risk_engine.position_sizing import risk_amount

logger = logging.getLogger(__name__)


def _recent_swing_levels(m5: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """Extract the most recent swing-high and swing-low price levels.

    Parameters
    ----------
    m5:
        Feature-enriched M5 DataFrame containing ``sr_resistance`` and
        ``sr_support`` columns (forward-filled swing levels produced by the
        structure feature engine).
    lookback:
        Number of recent bars to consider.

    Returns
    -------
    dict
        ``{"resistance": float, "support": float}`` with the most recent
        swing levels, or ``0.0`` when the columns are missing.
    """
    result = {"resistance": 0.0, "support": 0.0}

    if m5 is None or m5.empty:
        return result

    recent = m5.tail(lookback)

    if "sr_resistance" in recent.columns:
        resistance_vals = recent["sr_resistance"].dropna()
        if not resistance_vals.empty:
            result["resistance"] = float(resistance_vals.iloc[-1])

    if "sr_support" in recent.columns:
        support_vals = recent["sr_support"].dropna()
        if not support_vals.empty:
            result["support"] = float(support_vals.iloc[-1])

    return result


def apply_risk_parameters(
    featured_multi_tf: Dict[str, pd.DataFrame],
    signal: Dict,
) -> Dict:
    """Add stop-loss, take-profit, and risk metadata to a valid signal.

    Stop-loss placement
    -------------------
    * **BUY**  → SL is set below the most recent swing low.
    * **SELL** → SL is set above the most recent swing high.

    Take-profit placement
    ---------------------
    * TP = entry ± (|entry − SL| × :data:`~gold_quant_ai.config.DEFAULT_RR`)

    Additional fields added to *signal*
    ------------------------------------
    * ``sl``          — stop-loss price
    * ``tp``          — take-profit price
    * ``rr``          — realised risk-to-reward ratio
    * ``risk_pct``    — percentage of account risked
    * ``risk_amount`` — monetary risk in account currency

    Parameters
    ----------
    featured_multi_tf:
        Multi-timeframe feature dictionary; the ``"M5"`` key is used.
    signal:
        Valid signal dict from
        :func:`~gold_quant_ai.signal_engine.signal_generator.generate_signal`.

    Returns
    -------
    dict
        The *signal* dict updated with risk parameters.
    """
    signal = signal.copy()
    m5 = featured_multi_tf.get("M5")
    entry = signal.get("entry") or 0.0
    direction = signal.get("direction", "")

    levels = _recent_swing_levels(m5)
    support = levels["support"]
    resistance = levels["resistance"]

    sl: float = 0.0
    tp: float = 0.0

    if direction == "BUY":
        # SL below recent swing low; fall back to 1% below entry if unavailable
        sl = support if support and support < entry else round(entry * 0.99, 2)
        risk_pts = abs(entry - sl)
        tp = round(entry + risk_pts * DEFAULT_RR, 2)

    elif direction == "SELL":
        # SL above recent swing high; fall back to 1% above entry if unavailable
        sl = resistance if resistance and resistance > entry else round(entry * 1.01, 2)
        risk_pts = abs(sl - entry)
        tp = round(entry - risk_pts * DEFAULT_RR, 2)

    rr = DEFAULT_RR
    r_amount = risk_amount(ACCOUNT_BALANCE, RISK_PER_TRADE_PCT)

    signal.update(
        {
            "sl": round(sl, 2),
            "tp": tp,
            "rr": rr,
            "risk_pct": RISK_PER_TRADE_PCT,
            "risk_amount": r_amount,
        }
    )

    logger.info(
        "Risk params applied: %s entry=%.2f, sl=%.2f, tp=%.2f, rr=%.1f, risk=$%.2f",
        direction,
        entry,
        sl,
        tp,
        rr,
        r_amount,
    )
    return signal
