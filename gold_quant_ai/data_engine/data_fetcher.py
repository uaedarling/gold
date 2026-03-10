"""
data_engine/data_fetcher.py — OHLCV data acquisition for GOLD_QUANT_AI.

Attempts to fetch live bars from MetaTrader5 and falls back to CSV files
when MT5 is not available (e.g., in a Replit / cloud environment).
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_fetch_mt5(symbol: str, timeframe: str, bars: int = 600) -> Optional[pd.DataFrame]:
    """Attempt to fetch OHLCV bars from a running MetaTrader5 terminal.

    Parameters
    ----------
    symbol:
        The trading symbol, e.g. ``"XAUUSD"``.
    timeframe:
        A timeframe string such as ``"M5"``, ``"M15"``, ``"H1"``, or ``"H4"``.
    bars:
        Number of historical bars to request.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``[timestamp, open, high, low, close, volume]``
        or ``None`` when MT5 is unavailable.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        if not mt5.initialize():
            logger.debug("MT5 initialise() returned False — skipping live feed.")
            return None

        mt5_tf = tf_map.get(timeframe.upper())
        if mt5_tf is None:
            logger.warning("Unknown timeframe '%s' for MT5 — skipping.", timeframe)
            mt5.shutdown()
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    except Exception as exc:  # pragma: no cover — MT5 not available in CI
        logger.debug("MT5 fetch failed: %s", exc)
        return None


def _fetch_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.  The file must contain at
        minimum the columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the six OHLCV columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ohlcv_multi_timeframe(
    symbol: str,
    timeframes,
    csv_fallback_paths: Dict[str, str],
    bars: int = 600,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple timeframes.

    For each timeframe the function first tries the MetaTrader5 live feed.
    If MT5 is unavailable it falls back to the corresponding CSV path defined
    in *csv_fallback_paths*.

    Parameters
    ----------
    symbol:
        Trading symbol (e.g. ``"XAUUSD"``).
    timeframes:
        Iterable of timeframe strings, e.g. ``["M5", "M15", "H1", "H4"]``.
    csv_fallback_paths:
        Mapping of ``{timeframe: csv_path}`` used when MT5 is unavailable.
    bars:
        Number of bars to request from MT5.

    Returns
    -------
    dict
        ``{timeframe: pd.DataFrame}`` where each DataFrame has columns
        ``[timestamp, open, high, low, close, volume]``.
    """
    result: Dict[str, pd.DataFrame] = {}

    for tf in timeframes:
        df = _try_fetch_mt5(symbol, tf, bars)
        if df is not None and not df.empty:
            logger.info("Fetched %d bars for %s %s from MT5.", len(df), symbol, tf)
        else:
            fallback = csv_fallback_paths.get(tf)
            if not fallback:
                logger.warning("No CSV fallback configured for %s — skipping.", tf)
                continue
            try:
                df = _fetch_csv(fallback)
                logger.info("Loaded %d rows for %s %s from CSV '%s'.", len(df), symbol, tf, fallback)
            except Exception as exc:
                logger.error("Failed to load CSV for %s: %s", tf, exc)
                continue

        result[tf] = df

    return result
