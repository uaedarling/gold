"""
feature_engine/structure_features.py — Smart-money / market-structure feature builder.

Detects swing highs/lows, Break-of-Structure (BOS) events, liquidity sweeps,
and key support/resistance levels so the signal engine can apply smart-money
concepts to the filtered trade ideas.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SwingConfig:
    """Tunable parameters for swing-point and structure detection.

    Attributes
    ----------
    swing_lookback:
        Number of bars on each side required for a confirmed swing high/low.
    bos_lookback:
        Number of bars to look back when searching for the previous swing
        level that a Break-of-Structure must exceed.
    """

    swing_lookback: int = 3
    bos_lookback: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _swing_highs(high: pd.Series, lookback: int) -> pd.Series:
    """Return a boolean Series that is True at confirmed swing-high bars.

    A bar at index *i* is a swing high when ``high[i]`` is strictly greater
    than ``high[i-k]`` and ``high[i+k]`` for all k in 1..lookback.
    """
    result = pd.Series(False, index=high.index)
    for i in range(lookback, len(high) - lookback):
        window_left = high.iloc[i - lookback : i]
        window_right = high.iloc[i + 1 : i + lookback + 1]
        if high.iloc[i] > window_left.max() and high.iloc[i] > window_right.max():
            result.iloc[i] = True
    return result


def _swing_lows(low: pd.Series, lookback: int) -> pd.Series:
    """Return a boolean Series that is True at confirmed swing-low bars."""
    result = pd.Series(False, index=low.index)
    for i in range(lookback, len(low) - lookback):
        window_left = low.iloc[i - lookback : i]
        window_right = low.iloc[i + 1 : i + lookback + 1]
        if low.iloc[i] < window_left.min() and low.iloc[i] < window_right.min():
            result.iloc[i] = True
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_structure_features(df: pd.DataFrame, cfg: SwingConfig = None) -> pd.DataFrame:
    """Add market-structure columns to *df* and return it.

    Columns appended
    ----------------
    * ``swing_high``              — True when bar is a confirmed swing high
    * ``swing_low``               — True when bar is a confirmed swing low
    * ``hh``                      — Higher High (swing_high > previous swing_high)
    * ``hl``                      — Higher Low  (swing_low  > previous swing_low)
    * ``lh``                      — Lower High  (swing_high < previous swing_high)
    * ``ll``                      — Lower Low   (swing_low  < previous swing_low)
    * ``bos_up``                  — Bullish Break of Structure
    * ``bos_down``                — Bearish Break of Structure
    * ``liquidity_sweep_high``    — Candle swept above prior high but closed below it
    * ``liquidity_sweep_low``     — Candle swept below prior low but closed above it
    * ``sr_resistance``           — Most recent swing-high price level
    * ``sr_support``              — Most recent swing-low price level

    Parameters
    ----------
    df:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    cfg:
        Optional :class:`SwingConfig` instance; uses defaults when ``None``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the additional structure columns.
    """
    if cfg is None:
        cfg = SwingConfig()

    df = df.copy()
    n = len(df)

    # ------------------------------------------------------------------
    # Swing highs and lows
    # ------------------------------------------------------------------
    df["swing_high"] = _swing_highs(df["high"], cfg.swing_lookback)
    df["swing_low"] = _swing_lows(df["low"], cfg.swing_lookback)

    # ------------------------------------------------------------------
    # HH / HL / LH / LL  (compare each swing point to the previous one)
    # ------------------------------------------------------------------
    hh = pd.Series(False, index=df.index)
    hl = pd.Series(False, index=df.index)
    lh = pd.Series(False, index=df.index)
    ll = pd.Series(False, index=df.index)

    prev_sh_price: float = np.nan
    prev_sl_price: float = np.nan

    for i in range(n):
        if df["swing_high"].iloc[i]:
            cur = df["high"].iloc[i]
            if not np.isnan(prev_sh_price):
                if cur > prev_sh_price:
                    hh.iloc[i] = True
                else:
                    lh.iloc[i] = True
            prev_sh_price = cur

        if df["swing_low"].iloc[i]:
            cur = df["low"].iloc[i]
            if not np.isnan(prev_sl_price):
                if cur > prev_sl_price:
                    hl.iloc[i] = True
                else:
                    ll.iloc[i] = True
            prev_sl_price = cur

    df["hh"] = hh
    df["hl"] = hl
    df["lh"] = lh
    df["ll"] = ll

    # ------------------------------------------------------------------
    # Break of Structure
    # ------------------------------------------------------------------
    bos_up = pd.Series(False, index=df.index)
    bos_down = pd.Series(False, index=df.index)

    for i in range(cfg.bos_lookback, n):
        window_high = df["high"].iloc[i - cfg.bos_lookback : i]
        window_low = df["low"].iloc[i - cfg.bos_lookback : i]
        prev_high = window_high.max()
        prev_low = window_low.min()

        if df["close"].iloc[i] > prev_high:
            bos_up.iloc[i] = True
        if df["close"].iloc[i] < prev_low:
            bos_down.iloc[i] = True

    df["bos_up"] = bos_up
    df["bos_down"] = bos_down

    # ------------------------------------------------------------------
    # Liquidity sweeps
    # ------------------------------------------------------------------
    liq_sweep_high = pd.Series(False, index=df.index)
    liq_sweep_low = pd.Series(False, index=df.index)

    for i in range(1, n):
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        cur_high = df["high"].iloc[i]
        cur_low = df["low"].iloc[i]
        cur_close = df["close"].iloc[i]

        # Wick above previous high but close back below it
        if cur_high > prev_high and cur_close < prev_high:
            liq_sweep_high.iloc[i] = True

        # Wick below previous low but close back above it
        if cur_low < prev_low and cur_close > prev_low:
            liq_sweep_low.iloc[i] = True

    df["liquidity_sweep_high"] = liq_sweep_high
    df["liquidity_sweep_low"] = liq_sweep_low

    # ------------------------------------------------------------------
    # Support / Resistance levels (forward-fill most recent swing point)
    # ------------------------------------------------------------------
    df["sr_resistance"] = df["high"].where(df["swing_high"]).ffill()
    df["sr_support"] = df["low"].where(df["swing_low"]).ffill()

    return df
