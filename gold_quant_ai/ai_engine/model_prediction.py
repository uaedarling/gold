"""
ai_engine/model_prediction.py — XGBoost model loading and inference for GOLD_QUANT_AI.

Loads a pre-trained model from disk (or triggers training on first run) and
produces directional predictions together with calibrated confidence scores
for the most recent M5 bar.
"""

import logging
import os
import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier  # type: ignore

from gold_quant_ai.config import MODEL_PATH, AI_CONFIDENCE_THRESHOLD
from gold_quant_ai.ai_engine.model_training import (
    FEATURE_COLUMNS,
    train_model,
    save_model,
)

logger = logging.getLogger(__name__)

# Module-level model cache so repeated calls within the same process reuse
# the already-loaded model. A lock ensures thread safety.
_cached_model: Optional[XGBClassifier] = None
_model_lock = threading.Lock()


def _load_model() -> Optional[XGBClassifier]:
    """Load the XGBoost model from :data:`~gold_quant_ai.config.MODEL_PATH`.

    Returns
    -------
    XGBClassifier or None
        The loaded model, or ``None`` if no model file exists yet.
    """
    global _cached_model

    with _model_lock:
        if _cached_model is not None:
            return _cached_model

        if not os.path.exists(MODEL_PATH):
            logger.info("No saved model found at '%s'.", MODEL_PATH)
            return None

        try:
            model = XGBClassifier()
            model.load_model(MODEL_PATH)
            _cached_model = model
            logger.info("Model loaded from '%s'.", MODEL_PATH)
            return model
        except Exception as exc:
            logger.warning("Failed to load model from '%s': %s", MODEL_PATH, exc)
            return None


def load_or_train_and_predict(
    featured_multi_tf: Dict[str, pd.DataFrame],
) -> Dict:
    """Return AI prediction results for the latest M5 bar.

    If no saved model exists the function trains a new one using the M5
    DataFrame, saves it to disk, then immediately runs inference.

    Parameters
    ----------
    featured_multi_tf:
        Dictionary mapping timeframe strings to feature-enriched DataFrames
        (output of the full feature engine pipeline).  Must contain at least
        the ``"M5"`` key.

    Returns
    -------
    dict
        Keys:

        * ``prob_up``      — predicted probability of an upward move
        * ``prob_down``    — predicted probability of a downward move
        * ``confidence``   — ``max(prob_up, prob_down)``
        * ``direction``    — ``"BUY"``, ``"SELL"``, or ``None`` (below threshold)
    """
    global _cached_model

    # Default / error result
    empty: Dict = {
        "prob_up": 0.5,
        "prob_down": 0.5,
        "confidence": 0.5,
        "direction": None,
    }

    m5 = featured_multi_tf.get("M5")
    if m5 is None or m5.empty:
        logger.warning("M5 DataFrame missing — cannot run AI prediction.")
        return empty

    model = _load_model()

    if model is None:
        logger.info("Training new model on M5 data …")
        try:
            with _model_lock:
                model = train_model(m5)
                save_model(model)
                _cached_model = model
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            return empty

    # Select available feature columns from the latest non-NaN row
    available_cols = [c for c in FEATURE_COLUMNS if c in m5.columns]
    latest_row = m5[available_cols].dropna().iloc[[-1]] if not m5[available_cols].dropna().empty else None

    if latest_row is None or latest_row.empty:
        logger.warning("No valid feature row available for prediction.")
        return empty

    try:
        probs = model.predict_proba(latest_row)[0]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return empty

    # XGBoost binary: probs[0]=class 0 (down), probs[1]=class 1 (up)
    prob_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
    prob_down = 1.0 - prob_up
    confidence = max(prob_up, prob_down)

    direction: Optional[str] = None
    if confidence >= AI_CONFIDENCE_THRESHOLD:
        direction = "BUY" if prob_up >= prob_down else "SELL"

    return {
        "prob_up": round(prob_up, 4),
        "prob_down": round(prob_down, 4),
        "confidence": round(confidence, 4),
        "direction": direction,
    }
