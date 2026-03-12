import MetaTrader5 as mt5
import msvcrt
import pandas as pd
import numpy as np
import time
import os
import csv
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False


#############################################
# SETTINGS
#############################################

SYMBOL               = "XAUUSDm"
SCAN_INTERVAL        = 5
MAGIC                = 123456

# ── Risk ─────────────────────────────────────
RISK_PER_TRADE       = 0.005
RISK_RECOVERY        = 0.0025
MAX_LOT_CAP          = 3.0
MIN_LOT              = 0.01

# ── Circuit Breakers ─────────────────────────
MAX_DAILY_LOSS_PCT   = 0.03
MAX_CONSEC_LOSSES    = 2
COOLDOWN_MINUTES     = 30
ATR_SPIKE_MULT       = 3.0

# ── ATR Multipliers ──────────────────────────
SL_ATR_MULT          = 0.5
TP1_ATR_MULT         = 0.8
TP2_ATR_MULT         = 1.5
TRAIL_ATR_MULT       = 0.7
BE_ATR_MULT          = 0.5

# ── Partial Exit Sizes ───────────────────────
TP1_CLOSE_PCT        = 0.40
TP2_CLOSE_PCT        = 0.35

# ── Entry Filters ────────────────────────────
MAX_SPREAD           = 1.5

# ── Standard Signal Thresholds ───────────────
# FIX 1: Relaxed RSI thresholds (was 40 / 60)
RSI_BUY_MAX          = 48        # oversold buy
RSI_SELL_MIN         = 55        # overbought sell
MACD_BUY_CROSS       = 0.05      # meaningfully bullish
MACD_SELL_CROSS      = -0.05     # meaningfully bearish

# ── Divergence Signal Settings ───────────────
# FIX 2: Fixed RSI divergence logic (was inverted — both were 50)
# FIX 3: Widened MACD divergence thresholds (was 0.5 / -0.5)
# Bearish divergence: price higher, MACD lower = exhaustion sell
# Bullish divergence: price lower,  MACD higher = exhaustion buy
MACD_DIVERGE_BARS    = 5         # candles to look back for divergence
MACD_DIVERGE_WEAK    = 2.0       # FIX 3: was 0.5 — M5 MACD runs 1–2 normally on gold
MACD_DIVERGE_STRONG  = -2.0      # FIX 3: was -0.5
RSI_DIVERGE_SELL_MAX = 70        # FIX 2: RSI falling from overbought for bearish div (was 50)
RSI_DIVERGE_BUY_MIN  = 30        # FIX 2: RSI recovering from oversold for bullish div (was 50 — backwards)

# ── Multi-Timeframe Settings ─────────────────
TF_WEIGHTS = {
    "M1":  1,
    "M5":  2,
    "M15": 3,
    "H1":  4,
}
# FIX 4: Lowered score threshold to 4 (was 5) — M5+M15 = 5, M1+M5+M15 = 6
MIN_BUY_SCORE        = 4
MIN_SELL_SCORE       = 4

# ── Session Settings UTC ─────────────────────
LONDON_OPEN          = 7
NY_OPEN              = 12
NY_CLOSE             = 21
ASIA_OPEN            = 22
SESSION_CLOSE_BUFFER = 30

# Asia: replaced hard block with ATR minimum filter
# If ATR is high enough during Asia, entries are allowed
ASIA_MIN_ATR         = 6.0       # M5 ATR must exceed this to trade Asia session

SESSION_ATR_MULT = {
    "LONDON":   1.2,
    "NEW YORK": 1.0,
    "ASIA":     0.8,
    "OVERLAP":  1.1,
}

# ── News Filter UTC ──────────────────────────
NEWS_BLOCK_MINUTES   = 30
MANUAL_NEWS_BLOCKS   = [
    "13:30",
    "18:00",
    "14:00",
]

# ── Slippage ─────────────────────────────────
MAX_ACCEPTABLE_SLIP  = 2.0

# ── Missed Opportunity Log ───────────────────
MAX_MISSED_LOG       = 5

# ── EMA200 Tolerance Band ────────────────────
# FIX 5: Allow entries within this many points of EMA200 (was a hard block at 0)
EMA200_TOLERANCE     = 2.0

# ── Data Saving ──────────────────────────────
DATE_STR             = datetime.now().strftime("%Y%m%d")
LOG_FILE             = f"goldbot_{DATE_STR}.log"
TRADES_FILE          = f"goldbot_trades_{DATE_STR}.csv"
SIGNALS_FILE         = f"goldbot_signals_{DATE_STR}.csv"
MISSED_FILE          = f"goldbot_missed_{DATE_STR}.csv"
CANDLES_FILE         = f"goldbot_candles_{DATE_STR}.csv"

#############################################
# GLOBALS
#############################################

trailing_enabled     = True
trade_count          = 0
total_profit         = 0.0
wins                 = 0
losses               = 0
consec_losses        = 0
last_signal          = "NONE"
bot_status           = "RUNNING"
cooldown_until       = None
daily_start_balance  = None
tp_tracker           = {}
slippage_log         = []
atr_baseline         = None
missed_opportunities = []
open_trade_data      = {}

#############################################
# CSV INITIALISER
#############################################

def init_csv_files():
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "date","time","ticket","direction","lot",
                "entry_price","exit_price","sl","tp1","tp2",
                "profit","slippage","session","buy_score","sell_score",
                "signal_type","atr","rsi_m5","macd_m5","ema200_m5",
                "tp1_hit","tp2_hit","exit_reason","recovery_mode"
            ])

    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "date","time","direction","buy_score","sell_score",
                "signal_type","fired","block_reason",
                "price","spread","session","atr_baseline",
                "m1_rsi","m1_macd","m1_atr","m1_buy","m1_sell",
                "m5_rsi","m5_macd","m5_atr","m5_buy","m5_sell",
                "m15_rsi","m15_macd","m15_atr","m15_buy","m15_sell",
                "h1_rsi","h1_macd","h1_atr","h1_buy","h1_sell",
            ])

    if not os.path.exists(MISSED_FILE):
        with open(MISSED_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "date","time","direction","buy_score","sell_score",
                "signal_type","reason","price","spread","session",
                "m1_vote","m5_vote","m15_vote","h1_vote"
            ])

    if not os.path.exists(CANDLES_FILE):
        with open(CANDLES_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "date","time","tf",
                "open","high","low","close",
                "rsi","macd","atr","ema200",
                "buy_signal","sell_signal","signal_type"
            ])

#############################################
# CSV WRITERS
#############################################

def save_trade(ticket, direction, lot, entry_price, exit_price,
               sl, tp1, tp2, profit, slippage, session,
               buy_score, sell_score, signal_type, atr,
               rsi_m5, macd_m5, ema200_m5,
               tp1_hit, tp2_hit, exit_reason, recovery):
    with open(TRADES_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            ticket, direction, lot,
            entry_price, exit_price, sl, tp1, tp2,
            round(profit,2), round(slippage,4), session,
            buy_score, sell_score, signal_type,
            round(atr,4), round(rsi_m5,2),
            round(macd_m5,4), round(ema200_m5,2),
            tp1_hit, tp2_hit, exit_reason, recovery
        ])

def save_signal(direction, buy_score, sell_score, signal_type,
                fired, block_reason, price, spread, session, tf_data):
    m1  = tf_data.get("M1",  {})
    m5  = tf_data.get("M5",  {})
    m15 = tf_data.get("M15", {})
    h1  = tf_data.get("H1",  {})
    with open(SIGNALS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            direction, buy_score, sell_score, signal_type, fired, block_reason,
            price, spread, session,
            round(atr_baseline,4) if atr_baseline else "",
            m1.get("rsi",""),  m1.get("macd",""),  m1.get("atr",""),
            m1.get("buy",""),  m1.get("sell",""),
            m5.get("rsi",""),  m5.get("macd",""),  m5.get("atr",""),
            m5.get("buy",""),  m5.get("sell",""),
            m15.get("rsi",""), m15.get("macd",""), m15.get("atr",""),
            m15.get("buy",""), m15.get("sell",""),
            h1.get("rsi",""),  h1.get("macd",""),  h1.get("atr",""),
            h1.get("buy",""),  h1.get("sell",""),
        ])

def save_missed(direction, buy_score, sell_score, signal_type,
                reason, price, spread, session, tf_data):
    m1  = tf_data.get("M1",  {})
    m5  = tf_data.get("M5",  {})
    m15 = tf_data.get("M15", {})
    h1  = tf_data.get("H1",  {})
    with open(MISSED_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            direction, buy_score, sell_score, signal_type, reason,
            price, spread, session,
            m1.get("buy",""), m5.get("buy",""),
            m15.get("buy",""), h1.get("buy",""),
        ])

def save_candle_snapshot(tf_data):
    now_date = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    with open(CANDLES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for tf, d in tf_data.items():
            if not d:
                continue
            raw = get_data(tf, bars=2)
            if raw is not None:
                last = raw.iloc[-1]
                w.writerow([
                    now_date, now_time, tf,
                    round(last["open"],  2),
                    round(last["high"],  2),
                    round(last["low"],   2),
                    round(last["close"], 2),
                    d.get("rsi",        ""),
                    d.get("macd",       ""),
                    d.get("atr",        ""),
                    d.get("ema200",     ""),
                    d.get("buy",        ""),
                    d.get("sell",       ""),
                    d.get("signal_type","STANDARD"),
                ])
#############################################
# LOGGING
#############################################

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if COLOR:
        print(f"{Fore.CYAN}[{ts}]{Style.RESET_ALL} {msg}")
    else:
        print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def col(text, color):
    if COLOR:
        return f"{color}{text}{Style.RESET_ALL}"
    return str(text)

#############################################
# NON-BLOCKING INPUT
#############################################

def read_command():
    if msvcrt.kbhit():
        return msvcrt.getwch().lower()
    return None

#############################################
# SESSION
#############################################

def get_session():
    hour = datetime.utcnow().hour
    if LONDON_OPEN <= hour < NY_OPEN:
        return "LONDON"
    elif NY_OPEN <= hour < NY_CLOSE:
        return "NEW YORK"
    elif hour >= ASIA_OPEN or hour < LONDON_OPEN:
        return "ASIA"
    return "OVERLAP"

def session_allows_entry(current_atr=None):
    """
    Asia is no longer hard-blocked.
    Instead it checks live ATR — only allows entry
    if market is actually moving (ATR above minimum).
    """
    session = get_session()
    now_utc = datetime.utcnow()

    # Asia ATR filter — replaces hard block
    if session == "ASIA":
        if current_atr is not None and current_atr < ASIA_MIN_ATR:
            return False, f"ASIA LOW VOL | ATR={round(current_atr,2)} < min {ASIA_MIN_ATR}"

    # Pre-close buffer
    ny_close_today = now_utc.replace(hour=NY_CLOSE, minute=0, second=0)
    if now_utc >= ny_close_today - timedelta(minutes=SESSION_CLOSE_BUFFER):
        return False, f"PRE-CLOSE BUFFER — {SESSION_CLOSE_BUFFER}min before NY close"

    return True, "OK"

def get_session_atr_mult():
    return SESSION_ATR_MULT.get(get_session(), 1.0)

#############################################
# NEWS FILTER
#############################################

def is_news_window():
    now_mins = datetime.utcnow().hour * 60 + datetime.utcnow().minute
    for block_time in MANUAL_NEWS_BLOCKS:
        hh, mm    = map(int, block_time.split(":"))
        news_mins = hh * 60 + mm
        if abs(now_mins - news_mins) <= NEWS_BLOCK_MINUTES:
            return True, f"NEWS WINDOW around {block_time} UTC"
    return False, "OK"

#############################################
# DATA
#############################################

def get_data(tf, bars=200):
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
    }
    rates = mt5.copy_rates_from_pos(SYMBOL, tf_map[tf], 0, bars)
    if rates is None or len(rates) == 0:
        log(f"ERROR: No data for {tf}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df

#############################################
# FEATURES
#############################################

def add_features(df):
    df["rsi"]    = RSIIndicator(df["close"]).rsi()
    macd_obj     = MACD(df["close"])
    df["macd"]   = macd_obj.macd_diff()
    atr_obj      = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"]    = atr_obj.average_true_range()
    ema_obj      = EMAIndicator(df["close"], window=200)
    df["ema200"] = ema_obj.ema_indicator()
    return df

#############################################
# ATR SPIKE
#############################################

def update_atr_baseline(atr_series):
    global atr_baseline
    atr_baseline = atr_series.iloc[-21:-1].mean()

def atr_spike_detected(current_atr):
    if atr_baseline is None or atr_baseline == 0:
        return False
    if current_atr / atr_baseline >= ATR_SPIKE_MULT:
        log(f"ATR SPIKE | {round(current_atr,2)} vs baseline {round(atr_baseline,2)}")
        return True
    return False

#############################################
# SPREAD
#############################################

def get_spread():
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        return 999.0
    return round((tick.ask - tick.bid) / info.point, 2)

def spread_ok():
    spread = get_spread()
    if spread > MAX_SPREAD:
        log(f"SPREAD BLOCKED | {spread} > max {MAX_SPREAD}")
        return False
    return True

#############################################
# LOT SIZE
#############################################

def calculate_lot(entry, sl, recovery=False):
    info     = mt5.symbol_info(SYMBOL)
    balance  = mt5.account_info().balance
    distance = abs(entry - sl)
    if distance == 0:
        return MIN_LOT
    risk_pct = RISK_RECOVERY if recovery else RISK_PER_TRADE
    lot      = (balance * risk_pct) / (distance * info.trade_contract_size)
    return round(max(MIN_LOT, min(lot, MAX_LOT_CAP)), 2)

#############################################
# POSITION HELPERS
#############################################

def get_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return None
    for p in positions:
        if p.magic == MAGIC:
            return p
    return None

def has_open_position():
    return get_open_position() is not None

#############################################
# CIRCUIT BREAKERS
#############################################

def check_circuit_breakers():
    global bot_status, cooldown_until
    balance = mt5.account_info().balance
    if daily_start_balance and balance < daily_start_balance * (1 - MAX_DAILY_LOSS_PCT):
        bot_status = "HALTED"
        loss_pct   = round((1 - balance / daily_start_balance) * 100, 2)
        return False, f"DAILY LOSS LIMIT -{loss_pct}%"
    if cooldown_until and datetime.utcnow() < cooldown_until:
        remaining  = int((cooldown_until - datetime.utcnow()).total_seconds() / 60)
        bot_status = "COOLDOWN"
        return False, f"COOLDOWN {remaining}min remaining"
    if cooldown_until and datetime.utcnow() >= cooldown_until:
        cooldown_until = None
        bot_status     = "RUNNING"
    bot_status = "RUNNING"
    return True, "OK"

def is_recovery_mode():
    return consec_losses >= MAX_CONSEC_LOSSES

#############################################
# SLIPPAGE
#############################################

def log_slippage(expected, actual):
    slip = abs(actual - expected)
    slippage_log.append(slip)
    if slip > MAX_ACCEPTABLE_SLIP:
        log(f"SLIPPAGE WARN | Expected={expected} Got={actual} Slip={round(slip,2)}")
    return slip

#############################################
# DIVERGENCE DETECTION
#############################################

def check_bearish_divergence(df):
    """
    Bearish divergence: price made a higher high BUT
    MACD histogram is declining and weakening.
    Classic exhaustion signal — move about to reverse down.
    FIX 2: RSI threshold raised to 70 (was 50 — neutral, not overbought).
    FIX 3: MACD upper bound widened to 2.0 (was 0.5 — too tight for gold).
    """
    if len(df) < MACD_DIVERGE_BARS + 1:
        return False
    recent       = df.iloc[-(MACD_DIVERGE_BARS+1):]
    price_higher = recent["close"].iloc[-1] > recent["close"].iloc[0]
    macd_lower   = recent["macd"].iloc[-1]  < recent["macd"].iloc[0]
    # FIX 3: widened to MACD_DIVERGE_WEAK=2.0
    macd_weak    = 0 < recent["macd"].iloc[-1] < MACD_DIVERGE_WEAK
    # FIX 2: raised to RSI_DIVERGE_SELL_MAX=70
    rsi_falling  = recent["rsi"].iloc[-1] < RSI_DIVERGE_SELL_MAX
    return price_higher and macd_lower and macd_weak and rsi_falling

def check_bullish_divergence(df):
    """
    Bullish divergence: price made a lower low BUT
    MACD histogram is rising and strengthening.
    Classic exhaustion signal — move about to reverse up.
    FIX 2: RSI threshold lowered to 30 (was 50 — completely backwards,
            bullish divergence happens when oversold not at neutral).
    FIX 3: MACD lower bound widened to -2.0 (was -0.5 — too tight for gold).
    """
    if len(df) < MACD_DIVERGE_BARS + 1:
        return False
    recent       = df.iloc[-(MACD_DIVERGE_BARS+1):]
    price_lower  = recent["close"].iloc[-1] < recent["close"].iloc[0]
    macd_higher  = recent["macd"].iloc[-1]  > recent["macd"].iloc[0]
    # FIX 3: widened to MACD_DIVERGE_STRONG=-2.0
    macd_weak    = MACD_DIVERGE_STRONG < recent["macd"].iloc[-1] < 0
    # FIX 2: lowered to RSI_DIVERGE_BUY_MIN=30
    rsi_rising   = recent["rsi"].iloc[-1] > RSI_DIVERGE_BUY_MIN
    return price_lower and macd_higher and macd_weak and rsi_rising

#############################################
# MULTI-TIMEFRAME SIGNAL SCORER
#############################################

def score_timeframe(tf):
    """
    Returns (buy_score, sell_score, data_dict).
    Checks both standard RSI/MACD signals AND
    divergence exhaustion signals.
    signal_type recorded: STANDARD or DIVERGENCE
    FIX 1: RSI_BUY_MAX=48, RSI_SELL_MIN=55.
    FIX 5: EMA200 tolerance band applied — entries allowed
            within EMA200_TOLERANCE points of EMA200.
    """
    df = get_data(tf)
    if df is None:
        return 0, 0, {}

    df         = add_features(df)
    last       = df.iloc[-1]
    weight     = TF_WEIGHTS[tf]
    rsi        = last["rsi"]
    macd_val   = last["macd"]
    atr        = last["atr"]
    ema200     = last["ema200"]
    close      = last["close"]

    buy_score    = 0
    sell_score   = 0
    signal_type  = "NONE"

    # ── Standard signals ──────────────────────
    # FIX 1: RSI_BUY_MAX now 48 (was 40), RSI_SELL_MIN now 55 (was 60)
    # FIX 5: tolerance band on EMA200 check
    if rsi < RSI_BUY_MAX and macd_val > MACD_BUY_CROSS and close > (ema200 - EMA200_TOLERANCE):
        buy_score   = weight
        signal_type = "STANDARD"

    if rsi > RSI_SELL_MIN and macd_val < MACD_SELL_CROSS and close < (ema200 + EMA200_TOLERANCE):
        sell_score  = weight
        signal_type = "STANDARD"

    # ── Divergence signals (only if standard not already firing) ──
    if buy_score == 0 and check_bullish_divergence(df) and close < ema200:
        buy_score   = weight
        signal_type = "DIVERGENCE"

    if sell_score == 0 and check_bearish_divergence(df) and close < ema200:
        sell_score  = weight
        signal_type = "DIVERGENCE"

    data = {
        "tf":          tf,
        "close":       round(close,   2),
        "rsi":         round(rsi,     2),
        "macd":        round(macd_val,4),
        "atr":         round(atr,     2),
        "ema200":      round(ema200,  2),
        "buy":         buy_score  > 0,
        "sell":        sell_score > 0,
        "weight":      weight,
        "signal_type": signal_type,
    }

    return buy_score, sell_score, data


def scan_all_timeframes():
    total_buy    = 0
    total_sell   = 0
    tf_data      = {}
    signal_types = []

    for tf in TF_WEIGHTS.keys():
        b, s, d     = score_timeframe(tf)
        total_buy  += b
        total_sell += s
        tf_data[tf] = d
        if d.get("signal_type") not in ("NONE", ""):
            signal_types.append(d.get("signal_type"))

    # Determine dominant signal type for this cycle
    dominant = "DIVERGENCE" if "DIVERGENCE" in signal_types else \
               "STANDARD"   if "STANDARD"   in signal_types else "NONE"

    return total_buy, total_sell, tf_data, dominant


def log_missed_opportunity(direction, buy_score, sell_score,
                           signal_type, reason, tf_data):
    global missed_opportunities
    entry = {
        "time":        datetime.now().strftime("%H:%M:%S"),
        "direction":   direction,
        "buy_score":   buy_score,
        "sell_score":  sell_score,
        "signal_type": signal_type,
        "reason":      reason,
        "tfs":         {tf: d["buy"] if direction == "BUY" else d["sell"]
                        for tf, d in tf_data.items()},
    }
    missed_opportunities.insert(0, entry)
    missed_opportunities = missed_opportunities[:MAX_MISSED_LOG]
    score = buy_score if direction == "BUY" else sell_score
    log(f"MISSED OPP | {direction} | {signal_type} | Score={score} | {reason}")
    save_missed(
        direction, buy_score, sell_score, signal_type, reason,
        tf_data.get("M5", {}).get("close", ""),
        get_spread(), get_session(), tf_data
    )
#############################################
# DASHBOARD
#############################################

def dashboard(tf_data, buy_score, sell_score, spread, signal_type, current_atr):
    os.system("cls" if os.name == "nt" else "clear")

    session            = get_session()
    pos                = get_open_position()
    allowed, cb_reason = check_circuit_breakers()
    news_block, n_why  = is_news_window()
    sess_ok, s_why     = session_allows_entry(current_atr)
    recovery           = is_recovery_mode()

    m5     = tf_data.get("M5", {})
    price  = m5.get("close", 0)
    atr    = m5.get("atr",   0)
    ema200 = m5.get("ema200",0)

    status_color = Fore.GREEN if COLOR else ""
    if bot_status == "COOLDOWN":   status_color = Fore.YELLOW  if COLOR else ""
    if bot_status == "HALTED":     status_color = Fore.RED     if COLOR else ""
    if bot_status == "NEWS_BLOCK": status_color = Fore.MAGENTA if COLOR else ""

    sig_color = (Fore.MAGENTA if COLOR else "") if signal_type == "DIVERGENCE" \
                else (Fore.YELLOW if COLOR else "")

    print("=" * 64)
    print(col("             GOLD BOT V22 — LIVE DASHBOARD", Fore.YELLOW if COLOR else ""))
    print("=" * 64)
    print(f"  Status   : {col(bot_status, status_color)}"
          f"{'  [RECOVERY 0.25%]' if recovery else ''}")
    print(f"  Session  : {session}  |  ATR Mult: {get_session_atr_mult()}x"
          f"  |  Asia min ATR: {ASIA_MIN_ATR}")
    print(f"  Files    : {col('SAVING', Fore.GREEN if COLOR else '')} | "
          f"{TRADES_FILE}")

    if not sess_ok:  print(f"  {col('BLOCKED: '  + s_why,     Fore.RED     if COLOR else '')}")
    if news_block:   print(f"  {col('NEWS: '     + n_why,     Fore.MAGENTA if COLOR else '')}")
    if not allowed:  print(f"  {col('CIRCUIT: '  + cb_reason, Fore.RED     if COLOR else '')}")

    print("-" * 64)
    ema_col = (Fore.GREEN if COLOR else "") if price > ema200 else (Fore.RED if COLOR else "")
    atr_col = (Fore.RED   if COLOR else "") if atr_spike_detected(atr) else ""
    print(f"  Price    : {price}  |  EMA200: {col(str(ema200), ema_col)}"
          f"  {'[ABOVE]' if price > ema200 else '[BELOW]'}")
    print(f"  ATR(M5)  : {col(str(atr), atr_col)}"
          f"  (baseline: {round(atr_baseline,2) if atr_baseline else 'calc...'})"
          f"  |  Spread: "
          f"{col(str(spread), (Fore.GREEN if COLOR else '') if spread <= MAX_SPREAD else (Fore.RED if COLOR else ''))}")

    # ── Multi-TF Table ───────────────────────
    print("-" * 64)
    print(f"  {'TF':<6} {'Close':<10} {'RSI':<8} {'MACD':<10} {'ATR':<7} {'BUY':<5} {'SELL':<5} {'TYPE'}")
    print(f"  {'-'*58}")
    for tf in ["M1", "M5", "M15", "H1"]:
        d = tf_data.get(tf, {})
        if not d:
            print(f"  {tf:<6} N/A")
            continue
        buy_f  = col("YES", Fore.GREEN   if COLOR else "") if d["buy"]  else col("no",  Fore.RED if COLOR else "")
        sell_f = col("YES", Fore.GREEN   if COLOR else "") if d["sell"] else col("no",  Fore.RED if COLOR else "")
        rsi_c  = (Fore.GREEN if COLOR else "") if d["rsi"] < 40 else \
                 (Fore.RED   if COLOR else "") if d["rsi"] > 60 else ""
        macd_c = (Fore.GREEN if COLOR else "") if d["macd"] > 0 else (Fore.RED if COLOR else "")
        stype  = d.get("signal_type", "")
        st_col = (Fore.MAGENTA if COLOR else "") if stype == "DIVERGENCE" else ""
        print(f"  {tf:<6} {str(d['close']):<10}"
              f" {col(str(d['rsi']), rsi_c):<8}"
              f" {col(str(d['macd']), macd_c):<10}"
              f" {d['atr']:<7}"
              f" {buy_f:<5} {sell_f:<5}"
              f" {col(stype, st_col)}")

    # ── Score Bar ────────────────────────────
    print("-" * 64)
    max_s    = sum(TF_WEIGHTS.values())
    buy_bar  = "█" * buy_score  + "░" * (max_s - buy_score)
    sell_bar = "█" * sell_score + "░" * (max_s - sell_score)
    b_col    = (Fore.GREEN if COLOR else "") if buy_score  >= MIN_BUY_SCORE  else ""
    s_col    = (Fore.RED   if COLOR else "") if sell_score >= MIN_SELL_SCORE else ""

    print(f"  BUY  {col(str(buy_score).rjust(2), b_col)}/{max_s} [{buy_bar}]"
          f"  {'<<< SIGNAL' if buy_score >= MIN_BUY_SCORE else ''}")
    print(f"  SELL {col(str(sell_score).rjust(2), s_col)}/{max_s} [{sell_bar}]"
          f"  {'<<< SIGNAL' if sell_score >= MIN_SELL_SCORE else ''}")
    print(f"  Signal   : {col(last_signal, sig_color)}")
    if signal_type == "DIVERGENCE":
        print(f"  {col('  ** DIVERGENCE SIGNAL — momentum exhaustion detected **', Fore.MAGENTA if COLOR else '')}")

    # ── Open Position ────────────────────────
    print("-" * 64)
    if pos:
        ticket    = pos.ticket
        tp_info   = tp_tracker.get(ticket, {"tp1": False, "tp2": False})
        pnl_col   = (Fore.GREEN if COLOR else "") if pos.profit >= 0 else (Fore.RED if COLOR else "")
        dir_col   = (Fore.GREEN if COLOR else "") if pos.type == 0 else (Fore.RED if COLOR else "")
        direction = col("BUY",  dir_col) if pos.type == 0 else col("SELL", dir_col)
        tp1_st    = col("HIT",  Fore.GREEN if COLOR else "") if tp_info["tp1"] else "WAIT"
        tp2_st    = col("HIT",  Fore.GREEN if COLOR else "") if tp_info["tp2"] else "WAIT"
        td        = open_trade_data.get(ticket, {})
        print(f"  {direction}  Ticket={ticket}  Entry={pos.price_open}  SL={pos.sl}")
        print(f"  Vol={pos.volume}  P&L={col(str(round(pos.profit,2)), pnl_col)}"
              f"  TP1={tp1_st}  TP2={tp2_st}  Trail={'ON' if trailing_enabled else 'OFF'}")
        print(f"  Type={col(td.get('signal_type',''), sig_color)}")
        if slippage_log:
            avg_s  = round(sum(slippage_log[-10:]) / len(slippage_log[-10:]), 2)
            slip_c = (Fore.RED if COLOR else "") if avg_s > MAX_ACCEPTABLE_SLIP else (Fore.GREEN if COLOR else "")
            print(f"  Avg Slip: {col(str(avg_s)+' pips', slip_c)}")
    else:
        print("  No open position")

    # ── Stats ────────────────────────────────
    print("-" * 64)
    winrate = round((wins / trade_count * 100), 1) if trade_count > 0 else 0.0
    pnl_col = (Fore.GREEN if COLOR else "") if total_profit >= 0 else (Fore.RED if COLOR else "")
    dd_pct  = 0.0
    if daily_start_balance:
        dd_pct = round((1 - mt5.account_info().balance / daily_start_balance) * 100, 2)
    dd_col  = (Fore.RED if COLOR else "") if dd_pct > 1.5 else (Fore.GREEN if COLOR else "")
    print(f"  Trades={trade_count}  W={wins}  L={losses}  WR={winrate}%  ConsecL={consec_losses}")
    print(f"  P&L={col(str(round(total_profit,2)), pnl_col)}  "
          f"DailyDD={col(str(dd_pct)+'%', dd_col)} (limit {MAX_DAILY_LOSS_PCT*100}%)")

    # ── Missed Opps ──────────────────────────
    if missed_opportunities:
        print("-" * 64)
        print(f"  {col('MISSED OPPORTUNITIES', Fore.YELLOW if COLOR else '')}")
        for m in missed_opportunities:
            tf_hits = " ".join([f"{tf}:{'Y' if hit else 'N'}"
                                for tf, hit in m["tfs"].items()])
            score   = m["buy_score"] if m["direction"] == "BUY" else m["sell_score"]
            stype   = m.get("signal_type", "")
            print(f"  {m['time']} {m['direction']} [{stype}] "
                  f"Score={score} [{tf_hits}] -> {m['reason']}")

    print("-" * 64)
    print("  [c] Close  [p] Partial  [b] Breakeven  [t] Trail  [s] Stop")
    print("=" * 64)

#############################################
# OPEN TRADE
#############################################

def open_trade(direction, atr, price, ema200,
               buy_score, sell_score, signal_type, tf_data):
    global last_signal, open_trade_data

    if has_open_position():
        return

    # All gates — each logs missed opp if blocked
    allowed, cb_reason = check_circuit_breakers()
    if not allowed:
        log_missed_opportunity(direction, buy_score, sell_score,
                               signal_type, cb_reason, tf_data)
        save_signal(direction, buy_score, sell_score, signal_type,
                    False, cb_reason, price, get_spread(), get_session(), tf_data)
        return

    if not spread_ok():
        reason = f"SPREAD={get_spread()}"
        log_missed_opportunity(direction, buy_score, sell_score,
                               signal_type, reason, tf_data)
        save_signal(direction, buy_score, sell_score, signal_type,
                    False, reason, price, get_spread(), get_session(), tf_data)
        return

    news_block, n_why = is_news_window()
    if news_block:
        log_missed_opportunity(direction, buy_score, sell_score,
                               signal_type, n_why, tf_data)
        save_signal(direction, buy_score, sell_score, signal_type,
                    False, n_why, price, get_spread(), get_session(), tf_data)
        return

    sess_ok, s_why = session_allows_entry(atr)
    if not sess_ok:
        log_missed_opportunity(direction, buy_score, sell_score,
                               signal_type, s_why, tf_data)
        save_signal(direction, buy_score, sell_score, signal_type,
                    False, s_why, price, get_spread(), get_session(), tf_data)
        return

    if atr_spike_detected(atr):
        log_missed_opportunity(direction, buy_score, sell_score,
                               signal_type, "ATR SPIKE", tf_data)
        save_signal(direction, buy_score, sell_score, signal_type,
                    False, "ATR SPIKE", price, get_spread(), get_session(), tf_data)
        return

    # EMA200 filter — standard signals only
    # Divergence trades allowed on either side of EMA200
    # since they are momentum exhaustion, not trend-following
    # FIX 5: EMA200_TOLERANCE band — only block if price is more than
    #         EMA200_TOLERANCE points on the wrong side of EMA200
    if signal_type == "STANDARD":
        if direction == "BUY" and price < (ema200 - EMA200_TOLERANCE):
            reason = f"PRICE BELOW EMA200 by >{EMA200_TOLERANCE}pts"
            log_missed_opportunity(direction, buy_score, sell_score,
                                   signal_type, reason, tf_data)
            save_signal(direction, buy_score, sell_score, signal_type,
                        False, reason, price, get_spread(), get_session(), tf_data)
            return
        if direction == "SELL" and price > (ema200 + EMA200_TOLERANCE):
            reason = f"PRICE ABOVE EMA200 by >{EMA200_TOLERANCE}pts"
            log_missed_opportunity(direction, buy_score, sell_score,
                                   signal_type, reason, tf_data)
            save_signal(direction, buy_score, sell_score, signal_type,
                        False, reason, price, get_spread(), get_session(), tf_data)
            return

    tick      = mt5.symbol_info_tick(SYMBOL)
    entry     = tick.ask if direction == "BUY" else tick.bid
    sess_mult = get_session_atr_mult()
    adj_atr   = atr * sess_mult
    sl        = entry - (adj_atr * SL_ATR_MULT)  if direction == "BUY" \
                else entry + (adj_atr * SL_ATR_MULT)
    tp1       = entry + (adj_atr * TP1_ATR_MULT) if direction == "BUY" \
                else entry - (adj_atr * TP1_ATR_MULT)
    tp2       = entry + (adj_atr * TP2_ATR_MULT) if direction == "BUY" \
                else entry - (adj_atr * TP2_ATR_MULT)
    lot       = calculate_lot(entry, sl, recovery=is_recovery_mode())

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_BUY if direction == "BUY" \
                        else mt5.ORDER_TYPE_SELL,
        "price":        entry,
        "sl":           round(sl,  2),
        "tp":           round(tp1, 2),
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      f"GoldBotV22_{signal_type[:3]}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        slip        = log_slippage(entry, result.price)
        score_used  = buy_score if direction == "BUY" else sell_score
        last_signal = f"{direction} [{signal_type}] score={score_used}"
        log(f"ENTRY | {direction} | {signal_type} | Score={score_used}"
            f" | Lot={lot} | Entry={result.price}"
            f" | SL={round(sl,2)} | TP1={round(tp1,2)}"
            f"{' | RECOVERY' if is_recovery_mode() else ''}")

        m5d = tf_data.get("M5", {})
        open_trade_data[result.order] = {
            "ticket":      result.order,
            "direction":   direction,
            "lot":         lot,
            "entry":       result.price,
            "sl":          round(sl,  2),
            "tp1":         round(tp1, 2),
            "tp2":         round(tp2, 2),
            "slippage":    slip,
            "session":     get_session(),
            "buy_score":   buy_score,
            "sell_score":  sell_score,
            "signal_type": signal_type,
            "atr":         atr,
            "rsi_m5":      m5d.get("rsi",   0),
            "macd_m5":     m5d.get("macd",  0),
            "ema200_m5":   m5d.get("ema200",0),
            "recovery":    is_recovery_mode(),
        }
        save_signal(direction, buy_score, sell_score, signal_type,
                    True, "FIRED", result.price,
                    get_spread(), get_session(), tf_data)
    else:
        log(f"ORDER FAILED | retcode={result.retcode} | {result.comment}")

#############################################
# CLOSE TRADE
#############################################

def close_trade(pos, volume=None, exit_reason="MANUAL"):
    global trade_count, total_profit, wins, losses, consec_losses, cooldown_until

    vol   = volume if volume else pos.volume
    tick  = mt5.symbol_info_tick(SYMBOL)
    price = tick.bid if pos.type == 0 else tick.ask

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "position":  pos.ticket,
        "symbol":    SYMBOL,
        "volume":    round(vol, 2),
        "type":      mt5.ORDER_TYPE_SELL if pos.type == 0 \
                     else mt5.ORDER_TYPE_BUY,
        "price":     price,
        "deviation": 20,
        "magic":     MAGIC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        if not volume:
            trade_count  += 1
            total_profit += pos.profit
            tp_info       = tp_tracker.get(pos.ticket, {"tp1": False, "tp2": False})

            if pos.profit >= 0:
                wins          += 1
                consec_losses  = 0
                log(f"CLOSED WIN | +{round(pos.profit,2)} | {exit_reason}")
            else:
                losses        += 1
                consec_losses += 1
                log(f"CLOSED LOSS | {round(pos.profit,2)} | "
                    f"ConsecL={consec_losses} | {exit_reason}")
                if consec_losses >= MAX_CONSEC_LOSSES:
                    cooldown_until = datetime.utcnow() + timedelta(minutes=COOLDOWN_MINUTES)
                    log(f"COOLDOWN | Pausing {COOLDOWN_MINUTES}min")

            td = open_trade_data.get(pos.ticket, {})
            save_trade(
                ticket      = pos.ticket,
                direction   = td.get("direction",   ""),
                lot         = td.get("lot",          pos.volume),
                entry_price = td.get("entry",        pos.price_open),
                exit_price  = result.price,
                sl          = td.get("sl",           pos.sl),
                tp1         = td.get("tp1",          0),
                tp2         = td.get("tp2",          0),
                profit      = pos.profit,
                slippage    = td.get("slippage",     0),
                session     = td.get("session",      get_session()),
                buy_score   = td.get("buy_score",    0),
                sell_score  = td.get("sell_score",   0),
                signal_type = td.get("signal_type",  ""),
                atr         = td.get("atr",          0),
                rsi_m5      = td.get("rsi_m5",       0),
                macd_m5     = td.get("macd_m5",      0),
                ema200_m5   = td.get("ema200_m5",    0),
                tp1_hit     = tp_info["tp1"],
                tp2_hit     = tp_info["tp2"],
                exit_reason = exit_reason,
                recovery    = td.get("recovery",     False),
            )

            if pos.ticket in tp_tracker:
                del tp_tracker[pos.ticket]
            if pos.ticket in open_trade_data:
                del open_trade_data[pos.ticket]
    else:
        log(f"CLOSE FAILED | retcode={result.retcode}")

#############################################
# MODIFY SL
#############################################

def _modify_sl(pos, new_sl):
    result = mt5.order_send({
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol":   SYMBOL,
        "sl":       round(new_sl, 2),
        "tp":       pos.tp,
    })
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"SL MODIFY FAILED | retcode={result.retcode}")

#############################################
# BREAKEVEN
#############################################

def move_to_breakeven(pos, atr):
    tick       = mt5.symbol_info_tick(SYMBOL)
    is_buy     = pos.type == 0
    sess_atr   = atr * get_session_atr_mult()
    be_trigger = (pos.price_open + sess_atr * BE_ATR_MULT) if is_buy \
                 else (pos.price_open - sess_atr * BE_ATR_MULT)
    current    = tick.bid if is_buy else tick.ask
    if is_buy and current >= be_trigger and pos.sl < pos.price_open:
        _modify_sl(pos, pos.price_open + 0.10)
        log(f"BREAKEVEN SET | SL -> {round(pos.price_open+0.10,2)}")
        return True
    elif not is_buy and current <= be_trigger and pos.sl > pos.price_open:
        _modify_sl(pos, pos.price_open - 0.10)
        log(f"BREAKEVEN SET | SL -> {round(pos.price_open-0.10,2)}")
        return True
    return False

def be_is_set(pos):
    if pos.type == 0:
        return pos.sl >= pos.price_open
    return pos.sl <= pos.price_open and pos.sl > 0

#############################################
# TRAILING STOP
#############################################

def update_trail(pos, atr):
    if not trailing_enabled or not be_is_set(pos):
        return
    tick     = mt5.symbol_info_tick(SYMBOL)
    sess_atr = atr * get_session_atr_mult()
    trail    = sess_atr * TRAIL_ATR_MULT
    if pos.type == 0:
        new_sl = round(tick.bid - trail, 2)
        if new_sl > pos.sl:
            _modify_sl(pos, new_sl)
            log(f"TRAIL BUY | SL -> {new_sl}")
    else:
        new_sl = round(tick.ask + trail, 2)
        if new_sl < pos.sl or pos.sl == 0:
            _modify_sl(pos, new_sl)
            log(f"TRAIL SELL | SL -> {new_sl}")

#############################################
# SCALP TP MANAGER
#############################################

def manage_take_profits(pos, atr):
    global tp_tracker
    ticket   = pos.ticket
    tick     = mt5.symbol_info_tick(SYMBOL)
    is_buy   = pos.type == 0
    entry    = pos.price_open
    sess_atr = atr * get_session_atr_mult()
    if ticket not in tp_tracker:
        tp_tracker[ticket] = {"tp1": False, "tp2": False}
    tp1_price = (entry + sess_atr * TP1_ATR_MULT) if is_buy \
                else (entry - sess_atr * TP1_ATR_MULT)
    tp2_price = (entry + sess_atr * TP2_ATR_MULT) if is_buy \
                else (entry - sess_atr * TP2_ATR_MULT)
    current   = tick.bid if is_buy else tick.ask

    if not tp_tracker[ticket]["tp1"]:
        if (is_buy and current >= tp1_price) or \
           (not is_buy and current <= tp1_price):
            vol = max(round(pos.volume * TP1_CLOSE_PCT, 2), MIN_LOT)
            if vol <= pos.volume:
                close_trade(pos, volume=vol, exit_reason="TP1")
                tp_tracker[ticket]["tp1"] = True
                log(f"TP1 HIT | Closed {vol} lots at {round(current,2)}")

    elif not tp_tracker[ticket]["tp2"]:
        if (is_buy and current >= tp2_price) or \
           (not is_buy and current <= tp2_price):
            pos = get_open_position()
            if pos and pos.ticket == ticket:
                vol = max(round(pos.volume * (TP2_CLOSE_PCT / (1.0 - TP1_CLOSE_PCT)), 2), MIN_LOT)
                if vol <= pos.volume:
                    close_trade(pos, volume=vol, exit_reason="TP2")
                    tp_tracker[ticket]["tp2"] = True
                    log(f"TP2 HIT | Closed {vol} lots at {round(current,2)}")

    if tp_tracker.get(ticket, {}).get("tp2"):
        pos = get_open_position()
        if pos and pos.ticket == ticket:
            update_trail(pos, atr * 0.5)

#############################################
# MANUAL CONTROLS
#############################################

def manual_controls(cmd):
    global trailing_enabled
    if cmd == "s":
        return "STOP"
    pos = get_open_position()
    if not pos:
        log("No open position")
        return
    m5 = get_data("M5")
    if m5 is None:
        return
    atr = add_features(m5)["atr"].iloc[-1]
    if cmd == "c":
        close_trade(pos, exit_reason="MANUAL CLOSE")
    elif cmd == "p":
        vol = max(round(pos.volume / 2, 2), MIN_LOT)
        if vol < pos.volume:
            close_trade(pos, volume=vol, exit_reason="MANUAL PARTIAL")
            log(f"Partial close | {vol} lots")
        else:
            log("Partial close skipped — too small")
    elif cmd == "b":
        move_to_breakeven(pos, atr)
    elif cmd == "t":
        trailing_enabled = not trailing_enabled
        log(f"Trailing {'ENABLED' if trailing_enabled else 'DISABLED'}")

#############################################
# SIGNAL ENGINE
#############################################

def generate_signal():
    global last_signal

    buy_score, sell_score, tf_data, signal_type = scan_all_timeframes()

    m5_data = tf_data.get("M5", {})
    atr     = m5_data.get("atr",    0)
    price   = m5_data.get("close",  0)
    ema200  = m5_data.get("ema200", 0)
    spread  = get_spread()

    m5_raw = get_data("M5")
    if m5_raw is not None:
        update_atr_baseline(add_features(m5_raw)["atr"])

    # Save candle snapshot every cycle
    save_candle_snapshot(tf_data)

    dashboard(tf_data, buy_score, sell_score, spread, signal_type, atr)

    # Manage open position
    pos = get_open_position()
    if pos:
        manage_take_profits(pos, atr)
        move_to_breakeven(pos, atr)
        update_trail(pos, atr)
        return

    if buy_score >= MIN_BUY_SCORE:
        last_signal = f"BUY [{signal_type}] score={buy_score}"
        open_trade("BUY", atr, price, ema200,
                   buy_score, sell_score, signal_type, tf_data)

    elif sell_score >= MIN_SELL_SCORE:
        last_signal = f"SELL [{signal_type}] score={sell_score}"
        open_trade("SELL", atr, price, ema200,
                   buy_score, sell_score, signal_type, tf_data)

    else:
        last_signal = f"WAITING | BUY={buy_score} SELL={sell_score} need={MIN_BUY_SCORE}"

#############################################
# MAIN LOOP
#############################################

if not mt5.initialize():
    print("MT5 connection failed — check terminal is running")
    quit()

daily_start_balance = mt5.account_info().balance

init_csv_files()

log("Gold Bot V22 started")
log(f"Symbol={SYMBOL} | Balance={daily_start_balance} | Magic={MAGIC}")
log(f"Asia ATR filter={ASIA_MIN_ATR} | Divergence bars={MACD_DIVERGE_BARS}")
log(f"Files: {TRADES_FILE} | {SIGNALS_FILE} | {MISSED_FILE} | {CANDLES_FILE}")

while True:
    try:
        if bot_status == "HALTED":
            os.system("cls" if os.name == "nt" else "clear")
            print(col("BOT HALTED — Daily loss limit reached.",
                      Fore.RED if COLOR else ""))
            print("Press [s] to exit.")
            if read_command() == "s":
                break
            time.sleep(5)
            continue

        generate_signal()

        cmd = read_command()
        if cmd:
            result = manual_controls(cmd)
            if result == "STOP":
                log("Bot stopped by user")
                break

        time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log("Bot stopped via keyboard interrupt")
        break

mt5.shutdown()