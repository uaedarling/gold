"""
alerts/telegram_alert.py — Telegram notification and signal logging for GOLD_QUANT_AI.

Formats trade signals as human-readable Telegram messages and POSTs them to
the Bot API.  When Telegram credentials are absent the function is a safe
no-op so the rest of the system keeps running normally.  All signals are also
appended to a local CSV log for audit and backtesting purposes.
"""

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict

import requests  # type: ignore

from gold_quant_ai.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals_log.csv")

# Refresh interval (seconds) exposed so the dashboard HTML can use it
DASHBOARD_REFRESH_SECONDS = 15

_LOG_FIELDNAMES = [
    "timestamp",
    "direction",
    "entry",
    "sl",
    "tp",
    "rr",
    "risk_pct",
    "risk_amount",
    "trend_h4",
    "trend_h1",
    "ai_confidence",
    "prob_up",
    "prob_down",
    "reason",
]


def _format_message(signal: Dict) -> str:
    """Format a signal dict as a Telegram-ready text message.

    Parameters
    ----------
    signal:
        Signal dictionary produced by the signal and risk engines.

    Returns
    -------
    str
        A multi-line plain-text message suitable for sending via Telegram.
    """
    direction = signal.get("direction", "N/A")
    entry = signal.get("entry", 0.0)
    sl = signal.get("sl", 0.0)
    tp = signal.get("tp", 0.0)
    trend_h4 = signal.get("trend_h4", "N/A")
    trend_h1 = signal.get("trend_h1", "N/A")
    confidence = signal.get("ai_confidence", 0.0)

    emoji = "🟢" if direction == "BUY" else "🔴"

    return (
        f"{emoji} *GOLD SIGNAL DETECTED*\n\n"
        f"Direction: *{direction}*\n\n"
        f"Entry Price: *{entry:.2f}*\n"
        f"Stop Loss:  *{sl:.2f}*\n"
        f"Take Profit: *{tp:.2f}*\n\n"
        f"Trend Alignment:\n"
        f"  H4: {trend_h4}\n"
        f"  H1: {trend_h1}\n\n"
        f"AI Confidence: *{confidence:.1%}*"
    )


def send_telegram_alert(signal: Dict) -> None:
    """Send a formatted signal alert via the Telegram Bot API.

    If :data:`~gold_quant_ai.config.TELEGRAM_BOT_TOKEN` or
    :data:`~gold_quant_ai.config.TELEGRAM_CHAT_ID` are not set this function
    logs a debug message and returns immediately without raising an error.

    Parameters
    ----------
    signal:
        Valid signal dict (must contain at least ``direction``, ``entry``,
        ``sl``, ``tp``, ``trend_h4``, ``trend_h1``, and ``ai_confidence``).
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram credentials not configured — skipping alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": _format_message(signal),
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram alert sent (status %d).", resp.status_code)
    except requests.RequestException as exc:
        logger.error("Telegram alert failed: %s", exc)


def log_signal(signal: Dict) -> None:
    """Append a signal to the local CSV log file.

    Creates the file with a header row on first write.  Subsequent calls
    append rows without re-writing the header.

    Parameters
    ----------
    signal:
        Signal dict to persist.  Missing keys are stored as empty strings.
    """
    file_exists = os.path.isfile(_LOG_PATH)

    try:
        with open(_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()

            row = {k: signal.get(k, "") for k in _LOG_FIELDNAMES}
            if not row.get("timestamp"):
                row["timestamp"] = datetime.now(timezone.utc).isoformat()
            writer.writerow(row)

        logger.info("Signal logged to '%s'.", _LOG_PATH)
    except OSError as exc:
        logger.error("Failed to write signal log: %s", exc)
