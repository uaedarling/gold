"""
signal_engine/signal_generator.py — Full multi-timeframe signal generation pipeline.

Orchestrates the four-step trade qualification workflow:
  1. H4 macro bias via EMA-200.
  2. H1 intermediate trend structure confirmation.
  3. M15 pullback-to-EMA detection.
  4. M5 precision entry confirmation + AI confidence gate.

Only signals that pass every step are marked as valid for further processing
by the risk engine and alert system.
"""

import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from gold_quant_ai.config import AI_CONFIDENCE_THRESHOLD
from gold_quant_ai.signal_engine.trend_analysis import market_bias_h4, confirm_trend_h1
from gold_quant_ai.signal_engine.pullback_detector import detect_pullback_m15
from gold_quant_ai.signal_engine.entry_detector import detect_entry_m5

logger = logging.getLogger(__name__)


def generate_signal(
    featured_multi_tf: Dict[str, pd.DataFrame],
    ai: Dict,
) -> Dict:
    """Run the full 4-step signal generation workflow.

    Parameters
    ----------
    featured_multi_tf:
        Dictionary mapping timeframe strings to feature-enriched DataFrames.
        Must contain keys ``"M5"``, ``"M15"``, ``"H1"``, and ``"H4"``.
    ai:
        AI prediction result from
        :func:`~gold_quant_ai.ai_engine.model_prediction.load_or_train_and_predict`.
        Expected keys: ``prob_up``, ``prob_down``, ``confidence``,
        ``direction``.

    Returns
    -------
    dict
        Signal dictionary with keys:

        * ``valid``         — bool, True if all four criteria are met
        * ``direction``     — ``"BUY"``, ``"SELL"``, or ``None``
        * ``entry``         — current M5 close price (float)
        * ``timestamp``     — ISO-8601 UTC timestamp string
        * ``bias``          — macro bias string (``"BULLISH"``/``"BEARISH"``)
        * ``trend_h4``      — H4 EMA bias string
        * ``trend_h1``      — H1 structure trend string
        * ``ai_confidence`` — confidence score (float 0-1)
        * ``prob_up``       — upward probability (float)
        * ``prob_down``     — downward probability (float)
        * ``reason``        — human-readable string explaining the outcome
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    base: Dict = {
        "valid": False,
        "direction": None,
        "entry": None,
        "timestamp": now_utc,
        "bias": "NEUTRAL",
        "trend_h4": "UNKNOWN",
        "trend_h1": "UNKNOWN",
        "ai_confidence": ai.get("confidence", 0.0),
        "prob_up": ai.get("prob_up", 0.5),
        "prob_down": ai.get("prob_down", 0.5),
        "reason": "",
    }

    h4 = featured_multi_tf.get("H4")
    h1 = featured_multi_tf.get("H1")
    m15 = featured_multi_tf.get("M15")
    m5 = featured_multi_tf.get("M5")

    # ------------------------------------------------------------------
    # Step 1: H4 macro bias
    # ------------------------------------------------------------------
    trend_h4 = market_bias_h4(h4)
    base["trend_h4"] = trend_h4
    base["bias"] = trend_h4

    # ------------------------------------------------------------------
    # Step 2: H1 structure confirmation (must align with H4)
    # ------------------------------------------------------------------
    trend_h1 = confirm_trend_h1(h1)
    base["trend_h1"] = trend_h1

    if trend_h1 == "NEUTRAL":
        base["reason"] = "H1 trend is NEUTRAL — no clear structure."
        return base

    if trend_h4 != trend_h1:
        base["reason"] = (
            f"H4 bias ({trend_h4}) conflicts with H1 trend ({trend_h1}) — NO TRADE."
        )
        return base

    direction = "BUY" if trend_h4 == "BULLISH" else "SELL"

    # ------------------------------------------------------------------
    # Step 3: M15 pullback detection
    # ------------------------------------------------------------------
    pullback_ok = detect_pullback_m15(m15, trend_h4)
    if not pullback_ok:
        base["reason"] = f"No valid M15 pullback to EMA-50 detected for {direction}."
        return base

    # ------------------------------------------------------------------
    # Step 4a: M5 entry confirmation
    # ------------------------------------------------------------------
    entry_ok = detect_entry_m5(m5, direction)
    if not entry_ok:
        base["reason"] = f"M5 entry conditions not met for {direction}."
        return base

    # ------------------------------------------------------------------
    # Step 4b: AI confidence gate
    # ------------------------------------------------------------------
    ai_direction = ai.get("direction")
    ai_confidence = ai.get("confidence", 0.0)

    if ai_confidence < AI_CONFIDENCE_THRESHOLD:
        base["reason"] = (
            f"AI confidence {ai_confidence:.1%} below threshold "
            f"{AI_CONFIDENCE_THRESHOLD:.1%}."
        )
        return base

    if ai_direction is not None and ai_direction != direction:
        base["reason"] = (
            f"AI predicts {ai_direction} but technicals indicate {direction} — skipping."
        )
        return base

    # ------------------------------------------------------------------
    # Valid signal — populate entry price
    # ------------------------------------------------------------------
    entry_price = None
    if m5 is not None and not m5.empty and "close" in m5.columns:
        last_close = m5["close"].dropna()
        if not last_close.empty:
            entry_price = float(last_close.iloc[-1])

    base.update(
        {
            "valid": True,
            "direction": direction,
            "entry": entry_price,
            "reason": (
                f"All criteria met: H4={trend_h4}, H1={trend_h1}, "
                f"M15 pullback=✓, M5 entry=✓, AI={ai_confidence:.1%}."
            ),
        }
    )

    logger.info(
        "Signal generated: %s @ %.2f (AI confidence: %.1%%).",
        direction,
        entry_price or 0.0,
        ai_confidence,
    )
    return base
