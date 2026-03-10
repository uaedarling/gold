"""
ai_engine/model_training.py — XGBoost classifier training for GOLD_QUANT_AI.

Builds a training matrix from feature-enriched OHLCV data, trains an
XGBClassifier to predict whether price will be higher N bars in the future,
and persists the trained model to disk as JSON.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier  # type: ignore

from gold_quant_ai.config import MODEL_DIR, MODEL_PATH
from gold_quant_ai.data_engine.data_storage import ensure_dir

logger = logging.getLogger(__name__)

# Minimum number of training samples required after preprocessing
_MIN_TRAINING_SAMPLES = 50

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "ema_50",
    "ema_200",
    "rsi_14",
    "atr_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "candle_range",
    "avg_candle_range_20",
    "atr_expansion",
    "hh",
    "hl",
    "lh",
    "ll",
    "bos_up",
    "bos_down",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
]


# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------


def _build_labels(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """Build binary target labels: 1 if future close is higher, else 0.

    Parameters
    ----------
    df:
        Feature-enriched OHLCV DataFrame that must contain a ``close`` column.
    horizon:
        Number of bars forward to compare the current close against.

    Returns
    -------
    pd.Series
        Integer series of 0/1 labels aligned with *df*.
    """
    future_close = df["close"].shift(-horizon)
    label = (future_close > df["close"]).astype(int)
    return label


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------


def build_training_matrix(
    df: pd.DataFrame,
    horizon: int = 10,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct feature matrix *X* and label vector *y* ready for training.

    Parameters
    ----------
    df:
        Feature-enriched DataFrame (output of the feature engine).
    horizon:
        Forward-looking bar count used when building labels.

    Returns
    -------
    tuple
        ``(X, y)`` where *X* is a DataFrame of shape ``(n_samples, n_features)``
        and *y* is an integer Series of 0/1 labels.
    """
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_cols)
    if missing:
        logger.warning("Missing feature columns (will be skipped): %s", missing)

    labels = _build_labels(df, horizon=horizon)
    combined = df[available_cols].copy()
    combined["_label"] = labels

    # Drop rows where any feature or the label is NaN
    combined.dropna(inplace=True)

    X = combined[available_cols]
    y = combined["_label"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBClassifier on *df* and return the fitted model.

    Parameters
    ----------
    df:
        Feature-enriched OHLCV DataFrame produced by the feature engine.

    Returns
    -------
    XGBClassifier
        A trained XGBoost binary classifier.

    Raises
    ------
    ValueError
        If there are too few samples to train (< 50 after dropping NaN rows).
    """
    X, y = build_training_matrix(df)

    if len(X) < _MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Not enough training samples after preprocessing: {len(X)} rows. "
            f"Need at least {_MIN_TRAINING_SAMPLES}."
        )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    logger.info("XGBClassifier trained on %d samples (%d features).", len(X), X.shape[1])
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: XGBClassifier) -> None:
    """Save a trained XGBoost model to :data:`MODEL_PATH` as JSON.

    Parameters
    ----------
    model:
        Fitted :class:`~xgboost.XGBClassifier` instance to persist.
    """
    ensure_dir(MODEL_DIR)
    model.save_model(MODEL_PATH)
    logger.info("Model saved to '%s'.", MODEL_PATH)
