"""
GoldBot v12 — XAUUSDm MT5 Trading Bot
Automated gold scalper / swing bot using:
  • RSI, MACD, EMA200 trend filter
  • Multi-timeframe divergence detection (M1, M5, M15, H1)
  • ATR-based position sizing, TP/SL, trailing stop, breakeven
  • Session-based spread limits + signal-queue retry
  • Per-session unique CSV file timestamps
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import csv
import os
import time
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────
MAGIC          = 20260001
SYMBOL         = "XAUUSDm"
RISK_PER_TRADE = 0.01          # 1 % of account equity per trade
ATR_PERIOD     = 14
ATR_TF         = mt5.TIMEFRAME_M5
ATR_SL_MULT    = 1.5
ATR_TP_MULT    = 2.5
ATR_TRAIL_MULT = 1.0
EMA_PERIOD     = 200
RSI_PERIOD     = 14
RSI_OB         = 70
RSI_OS         = 30
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
SCAN_INTERVAL  = 5             # seconds between main loop iterations
MAX_POSITIONS  = 3             # max concurrent open trades
BREAKEVEN_R    = 1.0           # move SL to BE after this many R:R

# Session-aware spread limits (MT5 point units — XAUUSDm point=0.01)
SPREAD_LIMITS = {
    "LONDON":   20.0,
    "NEW YORK": 20.0,
    "ASIA":     50.0,
    "OVERLAP":  25.0,
}

# Signal queue
signal_queue     = []           # list of pending signals blocked by spread
SIGNAL_QUEUE_TTL = 60           # seconds a queued signal remains valid

# ─────────────────────────────────────────────────────────────────
# PER-SESSION FILE TIMESTAMPS
# Each bot run gets unique, never-overwritten file names.
# ─────────────────────────────────────────────────────────────────
SESSION_TS   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
LOG_FILE     = f"goldbot_{SESSION_TS}.log"
TRADES_FILE  = f"goldbot_trades_{SESSION_TS}.csv"
SIGNALS_FILE = f"goldbot_signals_{SESSION_TS}.csv"
MISSED_FILE  = f"goldbot_missed_{SESSION_TS}.csv"
CANDLES_FILE = f"goldbot_candles_{SESSION_TS}.csv"
SPREADS_FILE = f"goldbot_spreads_{SESSION_TS}.csv"

# ─────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────
MAX_DAILY_LOSS   = 0.05        # 5 % of equity
daily_loss_so_far = 0.0
circuit_broken   = False

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} UTC] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# CSV INITIALISATION
# ─────────────────────────────────────────────────────────────────
def init_csv_files() -> None:
    headers = {
        TRADES_FILE:  ["datetime", "ticket", "direction", "entry_price", "sl", "tp",
                       "lot_size", "atr", "session", "signal_type", "buy_score", "sell_score",
                       "pnl", "close_reason"],
        SIGNALS_FILE: ["datetime", "direction", "score", "session", "signal_type",
                       "price", "spread", "action"],
        MISSED_FILE:  ["datetime", "direction", "score", "session", "signal_type",
                       "price", "block_reason"],
        CANDLES_FILE: ["datetime", "tf", "open", "high", "low", "close",
                       "rsi", "macd", "macd_signal", "macd_hist"],
        SPREADS_FILE: ["datetime", "spread", "spread_limit", "session"],
    }
    for filepath, cols in headers.items():
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(cols)

# ─────────────────────────────────────────────────────────────────
# CSV WRITERS
# ─────────────────────────────────────────────────────────────────
def save_trade(ticket, direction, entry_price, sl, tp, lot_size, atr,
               session, signal_type, buy_score, sell_score,
               pnl=None, close_reason=None) -> None:
    with open(TRADES_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticket, direction, round(entry_price, 5), round(sl, 5), round(tp, 5),
            round(lot_size, 2), round(atr, 5),
            session, signal_type, buy_score, sell_score,
            round(pnl, 2) if pnl is not None else "",
            close_reason or "",
        ])


def save_signal(direction, score, session, signal_type, price, spread, action) -> None:
    with open(SIGNALS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            direction, score, session, signal_type,
            round(price, 5), round(spread, 2), action,
        ])


def save_missed(direction, score, session, signal_type, price, block_reason) -> None:
    with open(MISSED_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            direction, score, session, signal_type,
            round(price, 5), block_reason,
        ])


def save_candle(tf, candle, rsi, macd_val, macd_sig, macd_hist) -> None:
    with open(CANDLES_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tf,
            round(candle["open"], 5), round(candle["high"], 5),
            round(candle["low"],  5), round(candle["close"], 5),
            round(rsi,       2), round(macd_val,  5),
            round(macd_sig,  5), round(macd_hist, 5),
        ])


def save_spread(spread: float, spread_limit: float, session: str) -> None:
    with open(SPREADS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(spread, 2),
            round(spread_limit, 2),
            session,
        ])

# ─────────────────────────────────────────────────────────────────
# SESSION DETECTION
# ─────────────────────────────────────────────────────────────────
def get_session() -> str:
    hour = datetime.utcnow().hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 9:
        return "OVERLAP"
    if 9 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 22:
        return "NEW YORK"
    return "ASIA"

# ─────────────────────────────────────────────────────────────────
# MARKET DATA HELPERS
# ─────────────────────────────────────────────────────────────────
def get_spread() -> float:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 9999.0
    info = mt5.symbol_info(SYMBOL)
    point = info.point if info else 0.01
    return (tick.ask - tick.bid) / point


def get_price() -> tuple:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 0.0, 0.0
    return tick.bid, tick.ask


def get_candles(tf, count: int = 300) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, count)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# ─────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series) -> tuple:
    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd, sig, macd - sig


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

# ─────────────────────────────────────────────────────────────────
# SPREAD GATE  (returns tuple: ok, spread, limit)
# ─────────────────────────────────────────────────────────────────
def spread_ok(session=None) -> tuple:
    spread = get_spread()
    sess   = session or get_session()
    limit  = SPREAD_LIMITS.get(sess, 20.0)
    if spread > limit:
        log(f"SPREAD BLOCKED | {spread} > limit {limit} ({sess})")
        return False, spread, limit
    return True, spread, limit

# ─────────────────────────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────────────────────────
def calc_lot_size(sl_pips: float) -> float:
    account = mt5.account_info()
    if account is None:
        return 0.01
    equity   = account.equity
    risk_amt = equity * RISK_PER_TRADE
    info     = mt5.symbol_info(SYMBOL)
    if info is None:
        return 0.01
    pip_val  = info.trade_tick_value / info.trade_tick_size * info.point
    lot      = risk_amt / (sl_pips * pip_val) if sl_pips > 0 else 0.01
    lot_stepped = round(lot / info.volume_step) * info.volume_step
    lot         = max(info.volume_min, min(info.volume_max, lot_stepped))
    return lot

# ─────────────────────────────────────────────────────────────────
# DIVERGENCE DETECTION  (multi-timeframe)
# ─────────────────────────────────────────────────────────────────
TF_CONFIG = {
    "M1":  {"tf": mt5.TIMEFRAME_M1,  "weight": 1},
    "M5":  {"tf": mt5.TIMEFRAME_M5,  "weight": 2},
    "M15": {"tf": mt5.TIMEFRAME_M15, "weight": 3},
    "H1":  {"tf": mt5.TIMEFRAME_H1,  "weight": 4},
}


def detect_divergence(df: pd.DataFrame) -> dict:
    """Return {"buy": bool, "sell": bool} for the last candle."""
    if df is None or len(df) < 30:
        return {"buy": False, "sell": False}
    rsi  = calc_rsi(df["close"])
    lows  = df["low"]
    highs = df["high"]
    # Last 30 candles, look for swing pivots
    n  = len(df) - 1
    lb = 5
    result = {"buy": False, "sell": False}
    # Bullish divergence: price lower low, RSI higher low
    if n >= lb:
        prev_low_idx = lows.iloc[n - lb:n].idxmin()
        if lows.iloc[n] < lows.iloc[prev_low_idx] and rsi.iloc[n] > rsi.iloc[prev_low_idx]:
            result["buy"] = True
        # Bearish divergence: price higher high, RSI lower high
        prev_high_idx = highs.iloc[n - lb:n].idxmax()
        if highs.iloc[n] > highs.iloc[prev_high_idx] and rsi.iloc[n] < rsi.iloc[prev_high_idx]:
            result["sell"] = True
    return result


def multi_tf_signal() -> dict:
    """
    Aggregate divergence signals across all timeframes.
    Returns: {"buy_score": int, "sell_score": int, "tf_data": dict, "signal_type": str}
    """
    buy_score  = 0
    sell_score = 0
    tf_data    = {}
    for label, cfg in TF_CONFIG.items():
        df = get_candles(cfg["tf"])
        if df is None:
            continue
        rsi    = calc_rsi(df["close"])
        macd_v, macd_s, macd_h = calc_macd(df["close"])
        div    = detect_divergence(df)
        weight = cfg["weight"]
        tf_data[label] = {
            "rsi":       float(rsi.iloc[-1]),
            "macd":      float(macd_v.iloc[-1]),
            "macd_sig":  float(macd_s.iloc[-1]),
            "macd_hist": float(macd_h.iloc[-1]),
            "buy_div":   div["buy"],
            "sell_div":  div["sell"],
        }
        if div["buy"]:
            buy_score  += weight
        if div["sell"]:
            sell_score += weight
    signal_type = ""
    if buy_score > 0 and sell_score == 0:
        signal_type = "BUY DIVERGENCE"
    elif sell_score > 0 and buy_score == 0:
        signal_type = "SELL DIVERGENCE"
    elif buy_score > 0 and sell_score > 0:
        signal_type = "MIXED"
    return {
        "buy_score":   buy_score,
        "sell_score":  sell_score,
        "tf_data":     tf_data,
        "signal_type": signal_type,
    }

# ─────────────────────────────────────────────────────────────────
# OPEN TRADE
# ─────────────────────────────────────────────────────────────────
def open_trade(direction: str, atr: float, price: float, ema200: float,
               buy_score: int, sell_score: int,
               signal_type: str, tf_data: dict) -> None:
    global signal_queue, circuit_broken

    if circuit_broken:
        log(f"CIRCUIT BREAKER ACTIVE — trade skipped ({direction})")
        return

    session = get_session()

    # Count open positions
    positions = mt5.positions_get(symbol=SYMBOL) or []
    if len(positions) >= MAX_POSITIONS:
        log(f"MAX POSITIONS ({MAX_POSITIONS}) reached — skipping {direction}")
        return

    # Trend filter
    trend_ok = (direction == "BUY"  and price > ema200) or \
               (direction == "SELL" and price < ema200)
    if not trend_ok:
        block = f"TREND_FILTER | EMA200={round(ema200, 5)}"
        score = buy_score if direction == "BUY" else sell_score
        save_missed(direction, score, session, signal_type, price, block)
        log(f"SIGNAL MISSED | {direction} | {block}")
        return

    # Spread gate
    ok, spread, limit = spread_ok(session)
    if not ok:
        score = buy_score if direction == "BUY" else sell_score
        block = f"SPREAD={spread}"
        save_missed(direction, score, session, signal_type, price, block)
        log(f"SIGNAL MISSED | {direction} | {block} > limit {limit} — queuing for retry")
        # Queue signal for retry when spread normalises
        signal_queue.append({
            "direction":   direction,
            "atr":         atr,
            "price":       price,
            "ema200":      ema200,
            "buy_score":   buy_score,
            "sell_score":  sell_score,
            "signal_type": signal_type,
            "tf_data":     tf_data,
            "queued_at":   datetime.utcnow(),
        })
        return

    # SL / TP
    sl_dist = atr * ATR_SL_MULT
    tp_dist = atr * ATR_TP_MULT
    if direction == "BUY":
        sl = price - sl_dist
        tp = price + tp_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl = price + sl_dist
        tp = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL

    lot = calc_lot_size(sl_dist)
    if lot <= 0:
        log(f"LOT SIZE = 0 — skipping {direction}")
        return

    req = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      SYMBOL,
        "volume":      lot,
        "type":        order_type,
        "price":       price,
        "sl":          round(sl, 5),
        "tp":          round(tp, 5),
        "deviation":   10,
        "magic":       MAGIC,
        "comment":     f"GoldBot_{signal_type}",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        ticket = result.order
        log(f"TRADE OPENED | {direction} | ticket={ticket} | lot={lot} | sl={round(sl,5)} | tp={round(tp,5)} | spread={spread}")
        score = buy_score if direction == "BUY" else sell_score
        save_trade(ticket, direction, price, sl, tp, lot, atr,
                   session, signal_type, buy_score, sell_score)
        save_signal(direction, score, session, signal_type, price, spread, "OPENED")
    else:
        err = result.retcode if result else "None"
        log(f"ORDER FAILED | {direction} | retcode={err}")

# ─────────────────────────────────────────────────────────────────
# SIGNAL QUEUE DRAIN  (retry signals blocked by spread)
# ─────────────────────────────────────────────────────────────────
def drain_signal_queue() -> None:
    global signal_queue
    current_utc   = datetime.utcnow()
    still_pending = []
    for sig in signal_queue:
        age = (current_utc - sig["queued_at"]).total_seconds()
        if age > SIGNAL_QUEUE_TTL:
            log(f"SIGNAL QUEUE EXPIRED | {sig['direction']} | age={int(age)}s")
            continue
        ok, spread, limit = spread_ok(get_session())
        if ok:
            log(f"SIGNAL QUEUE RETRY | {sig['direction']} | spread normalised to {spread}")
            open_trade(
                sig["direction"], sig["atr"], sig["price"], sig["ema200"],
                sig["buy_score"], sig["sell_score"],
                sig["signal_type"], sig["tf_data"],
            )
        else:
            still_pending.append(sig)
    signal_queue = still_pending

# ─────────────────────────────────────────────────────────────────
# TRAILING STOP & BREAKEVEN MANAGEMENT
# ─────────────────────────────────────────────────────────────────
def manage_positions() -> None:
    positions = mt5.positions_get(symbol=SYMBOL) or []
    for pos in positions:
        if pos.magic != MAGIC:
            continue
        tick     = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            continue
        current  = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        df_atr   = get_candles(ATR_TF)
        if df_atr is None:
            continue
        atr      = calc_atr(df_atr)
        trail    = atr * ATR_TRAIL_MULT

        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = current - trail
            # Breakeven
            if current >= pos.price_open + atr * BREAKEVEN_R and pos.sl < pos.price_open:
                new_sl = max(new_sl, pos.price_open)
            if new_sl > pos.sl + 0.001:
                _modify_sl(pos, round(new_sl, 5))
        else:
            new_sl = current + trail
            if current <= pos.price_open - atr * BREAKEVEN_R and pos.sl > pos.price_open:
                new_sl = min(new_sl, pos.price_open)
            if new_sl < pos.sl - 0.001:
                _modify_sl(pos, round(new_sl, 5))


def _modify_sl(pos, new_sl: float) -> None:
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol":   SYMBOL,
        "sl":       new_sl,
        "tp":       pos.tp,
        "magic":    MAGIC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"SL MODIFIED | ticket={pos.ticket} | new_sl={new_sl}")

# ─────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER  (daily loss guard)
# ─────────────────────────────────────────────────────────────────
def check_circuit_breaker() -> None:
    global circuit_broken, daily_loss_so_far
    account = mt5.account_info()
    if account is None:
        return
    # Floating P&L of open positions
    positions = mt5.positions_get(symbol=SYMBOL) or []
    floating  = sum(p.profit for p in positions if p.magic == MAGIC)
    if floating < 0 and abs(floating) / account.equity > MAX_DAILY_LOSS:
        if not circuit_broken:
            log(f"CIRCUIT BREAKER TRIPPED | floating loss {floating:.2f} > {MAX_DAILY_LOSS*100:.0f}% equity")
        circuit_broken = True
    else:
        circuit_broken = False

# ─────────────────────────────────────────────────────────────────
# DASHBOARD  (terminal summary every cycle)
# ─────────────────────────────────────────────────────────────────
def dashboard() -> None:
    account = mt5.account_info()
    if account is None:
        return
    session          = get_session()
    ok, spread, limit = spread_ok(session)
    sess_spread_limit = SPREAD_LIMITS.get(session, 20.0)
    df_atr           = get_candles(ATR_TF)
    atr_val          = calc_atr(df_atr) if df_atr is not None else 0.0
    positions        = mt5.positions_get(symbol=SYMBOL) or []
    open_pos         = [p for p in positions if p.magic == MAGIC]
    floating         = sum(p.profit for p in open_pos)

    print("\n" + "=" * 62)
    print(f"  GoldBot v12 | {SYMBOL} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Session     : {session}")
    print(f"  Balance     : {account.balance:.2f} | Equity: {account.equity:.2f}")
    print(f"  Floating P&L: {floating:.2f}")
    print(f"  ATR(M5)     : {round(atr_val,5)} | Spread: {spread} / limit {sess_spread_limit} ({'OK' if ok else 'HIGH'})")
    print(f"  Open trades : {len(open_pos)} / {MAX_POSITIONS}")
    print(f"  Queue       : {len(signal_queue)} pending signal(s)")
    print(f"  Circuit Brk : {'ACTIVE' if circuit_broken else 'OK'}")
    print("=" * 62)

# ─────────────────────────────────────────────────────────────────
# MAIN SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────
def generate_signal() -> None:
    # Try to retry any spread-blocked signals first
    drain_signal_queue()

    session = get_session()

    # Get M5 candles for ATR and EMA200
    df_m5 = get_candles(mt5.TIMEFRAME_M5, 300)
    if df_m5 is None or len(df_m5) < EMA_PERIOD + 10:
        log("Insufficient M5 data — skipping cycle")
        return

    atr    = calc_atr(df_m5)
    ema200 = float(calc_ema(df_m5["close"], EMA_PERIOD).iloc[-1])
    bid, ask = get_price()
    price  = bid   # use bid for BUY check, will be overridden by ask in order

    # Log spread every cycle
    ok, spread, limit = spread_ok(session)
    save_spread(spread, limit, session)

    # Multi-timeframe divergence
    sig = multi_tf_signal()
    buy_score  = sig["buy_score"]
    sell_score = sig["sell_score"]
    signal_type = sig["signal_type"]
    tf_data    = sig["tf_data"]

    MIN_SCORE = 3   # minimum confluence score to act on a signal

    if buy_score >= MIN_SCORE and sell_score == 0:
        direction = "BUY"
        save_signal(direction, buy_score, session, signal_type, ask, spread, "PENDING")
        open_trade(direction, atr, ask, ema200, buy_score, sell_score, signal_type, tf_data)
    elif sell_score >= MIN_SCORE and buy_score == 0:
        direction = "SELL"
        save_signal(direction, sell_score, session, signal_type, bid, spread, "PENDING")
        open_trade(direction, atr, bid, ema200, buy_score, sell_score, signal_type, tf_data)

# ─────────────────────────────────────────────────────────────────
# MT5 INITIALISATION
# ─────────────────────────────────────────────────────────────────
def init_mt5() -> bool:
    if not mt5.initialize():
        log(f"MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        log(f"Symbol {SYMBOL} not found")
        mt5.shutdown()
        return False
    if not info.visible:
        mt5.symbol_select(SYMBOL, True)
    log(f"MT5 connected | {SYMBOL} | point={info.point}")
    return True

# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    log("=" * 62)
    log(f"GoldBot v12 starting — session {SESSION_TS}")
    log(f"Symbol: {SYMBOL} | Magic: {MAGIC} | Risk: {RISK_PER_TRADE*100:.1f}%")
    log(f"Spread limits: Asia={SPREAD_LIMITS.get('ASIA', 50.0)} | London={SPREAD_LIMITS.get('LONDON', 20.0)} | NY={SPREAD_LIMITS.get('NEW YORK', 20.0)}")
    log(f"Signal queue TTL={SIGNAL_QUEUE_TTL}s (retry on spread normalization)")
    log(f"Files: {LOG_FILE} | {TRADES_FILE} | {SIGNALS_FILE} | {MISSED_FILE} | {CANDLES_FILE} | {SPREADS_FILE}")
    log("=" * 62)

    if not init_mt5():
        return

    init_csv_files()

    try:
        while True:
            try:
                check_circuit_breaker()
                manage_positions()
                generate_signal()
                dashboard()
            except Exception as exc:
                log(f"CYCLE ERROR | {exc}")
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        log("GoldBot stopped by user")
    finally:
        mt5.shutdown()
        log("MT5 disconnected")


if __name__ == "__main__":
    main()
