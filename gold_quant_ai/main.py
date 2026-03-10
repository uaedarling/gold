"""
GOLD_QUANT_AI - Main runner

Runs continuous multi-timeframe analysis + AI prediction + signal generation,
then pushes Telegram alerts and serves a Flask dashboard.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from gold_quant_ai.config import (
    LOOP_INTERVAL_SECONDS,
    SYMBOL,
    TIMEFRAMES,
    CSV_FALLBACK_PATHS,
    ENABLE_DASHBOARD,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
)
from gold_quant_ai.data_engine.data_fetcher import fetch_ohlcv_multi_timeframe
from gold_quant_ai.feature_engine.indicator_features import add_indicator_features
from gold_quant_ai.feature_engine.structure_features import add_structure_features
from gold_quant_ai.ai_engine.model_prediction import load_or_train_and_predict
from gold_quant_ai.signal_engine.signal_generator import generate_signal
from gold_quant_ai.risk_engine.tp_sl_calculator import apply_risk_parameters
from gold_quant_ai.alerts.telegram_alert import send_telegram_alert, log_signal
from gold_quant_ai.dashboard.web_dashboard import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _prepare_timeframe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise and sort a raw OHLCV DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame with a ``timestamp`` column.

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted DataFrame ready for the feature engine.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_once(state: dict) -> dict:
    """Execute one full analysis cycle and update *state* in-place.

    Workflow
    --------
    1. Fetch OHLCV data for all configured timeframes.
    2. Compute technical indicators and structure features.
    3. Run AI model prediction on the latest M5 bar.
    4. Generate a trading signal via the 4-step workflow.
    5. If the signal is valid: apply risk parameters, send Telegram alert,
       append to signal log.
    6. Update the shared state dictionary for the dashboard.

    Parameters
    ----------
    state:
        Mutable dictionary shared between the analysis loop and the dashboard.

    Returns
    -------
    dict
        The updated *state* dict.
    """
    data = fetch_ohlcv_multi_timeframe(
        symbol=SYMBOL,
        timeframes=TIMEFRAMES,
        csv_fallback_paths=CSV_FALLBACK_PATHS,
    )

    featured: dict = {}
    for tf, df in data.items():
        df = _prepare_timeframe_df(df)
        df = add_indicator_features(df)
        df = add_structure_features(df)
        featured[tf] = df

    ai_result = load_or_train_and_predict(featured)
    signal = generate_signal(featured, ai_result)

    if signal.get("valid"):
        signal = apply_risk_parameters(featured, signal)
        send_telegram_alert(signal)
        log_signal(signal)

    state["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    state["market_bias"] = signal.get("bias", "NEUTRAL")
    state["trend_h4"] = signal.get("trend_h4", "UNKNOWN")
    state["trend_h1"] = signal.get("trend_h1", "UNKNOWN")
    state["latest_signal"] = signal
    state["ai"] = ai_result
    state["last_error"] = None
    return state


def analysis_loop(shared_state: dict) -> None:
    """Continuously run :func:`run_once` every :data:`~gold_quant_ai.config.LOOP_INTERVAL_SECONDS`.

    Exceptions are caught so the loop never stops due to a transient error.

    Parameters
    ----------
    shared_state:
        Mutable dictionary shared with the Flask dashboard thread.
    """
    while True:
        try:
            run_once(shared_state)
        except Exception as exc:  # noqa: BLE001
            # Don't swallow SystemExit / KeyboardInterrupt
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            shared_state["last_error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Error in analysis loop: %s", exc)
        time.sleep(LOOP_INTERVAL_SECONDS)


def main() -> None:
    """Entry point — start the analysis loop and (optionally) the Flask dashboard."""
    shared_state: dict = {
        "last_run_utc": None,
        "last_error": None,
        "market_bias": "NEUTRAL",
        "trend_h4": "UNKNOWN",
        "trend_h1": "UNKNOWN",
        "latest_signal": {},
        "ai": {},
    }

    logger.info("Starting GOLD_QUANT_AI analysis loop (interval=%ds).", LOOP_INTERVAL_SECONDS)
    t = threading.Thread(target=analysis_loop, args=(shared_state,), daemon=True)
    t.start()

    if ENABLE_DASHBOARD:
        logger.info("Starting dashboard on %s:%d …", DASHBOARD_HOST, DASHBOARD_PORT)
        app = create_app(shared_state)
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
    else:
        logger.info("Dashboard disabled — running headless.")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
