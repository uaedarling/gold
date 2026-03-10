"""
config.py — Central configuration for GOLD_QUANT_AI.

All settings are loaded from environment variables with sensible defaults so the
system can run in any environment (Replit, local, Docker, etc.) without modifying
source code.
"""

import os

# ---------------------------------------------------------------------------
# Market / data settings
# ---------------------------------------------------------------------------

SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAMES = os.getenv("TIMEFRAMES", "M5,M15,H1,H4").split(",")

# Loop interval for the main analysis loop (seconds)
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))

# CSV fallback paths used when MetaTrader5 is unavailable
_BASE_CSV = os.path.join(os.path.dirname(__file__), "data", "historical_data.csv")
CSV_FALLBACK_PATHS = {
    "M5": os.getenv("CSV_M5", _BASE_CSV),
    "M15": os.getenv("CSV_M15", _BASE_CSV),
    "H1": os.getenv("CSV_H1", _BASE_CSV),
    "H4": os.getenv("CSV_H4", _BASE_CSV),
}

# ---------------------------------------------------------------------------
# Telegram alert settings
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# AI / model settings
# ---------------------------------------------------------------------------

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "ai_engine", "models"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "xgb_model.json"))
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.65"))

# ---------------------------------------------------------------------------
# Risk management settings
# ---------------------------------------------------------------------------

DEFAULT_RR = float(os.getenv("DEFAULT_RR", "2.0"))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

# ---------------------------------------------------------------------------
# Dashboard settings
# ---------------------------------------------------------------------------

ENABLE_DASHBOARD = os.getenv("ENABLE_DASHBOARD", "true").lower() in ("1", "true", "yes")
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "15"))

# ---------------------------------------------------------------------------
# Signal engine settings
# ---------------------------------------------------------------------------

# Maximum percentage distance from EMA-50 for a pullback to be considered valid
EMA_PROXIMITY_PCT = float(os.getenv("EMA_PROXIMITY_PCT", "0.15"))
