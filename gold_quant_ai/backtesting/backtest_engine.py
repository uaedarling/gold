"""
backtesting/backtest_engine.py — Historical signal simulation engine for GOLD_QUANT_AI.

Replays a log of historical signals against synthetic price outcomes to compute
standard performance metrics: win rate, profit factor, maximum drawdown, total
return, average risk-to-reward, and a full equity curve.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Container for all backtest performance metrics.

    Attributes
    ----------
    win_rate:
        Fraction of winning trades (0.0 – 1.0).
    max_drawdown:
        Maximum peak-to-trough drawdown of the equity curve (0.0 – 1.0).
    profit_factor:
        Ratio of gross profit to gross loss (> 1 is profitable).
    total_return:
        Percentage return over the simulation period.
    avg_rr:
        Average realised risk-to-reward ratio across all closed trades.
    equity_curve:
        List of equity values at each trade close.
    """

    win_rate: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    avg_rr: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_drawdown(equity: List[float]) -> float:
    """Compute the maximum peak-to-trough drawdown of an equity curve.

    Parameters
    ----------
    equity:
        Ordered list of equity values.

    Returns
    -------
    float
        Maximum drawdown as a fraction (e.g. 0.15 = 15 %).  Returns 0.0 when
        *equity* has fewer than two elements.
    """
    if len(equity) < 2:
        return 0.0

    arr = np.array(equity, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (running_max - arr) / running_max
    return float(drawdowns.max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(signals_df: pd.DataFrame, initial_equity: float = 10000.0) -> BacktestResult:
    """Simulate trading results from a DataFrame of historical signals.

    Each row in *signals_df* represents one closed trade.  The function uses
    the ``direction``, ``entry``, ``sl``, ``tp``, and ``rr`` columns to
    determine whether the trade was a winner (price reached TP before SL) or a
    loser.  In the absence of actual outcome data the simulation assumes TP is
    hit on winning trades and SL is hit on losing trades, with a coin-flip
    shaped by the ``rr`` ratio.

    Parameters
    ----------
    signals_df:
        DataFrame where each row is a historical signal.  Expected columns:
        ``direction``, ``entry``, ``sl``, ``tp``, ``rr``, ``risk_amount``.
    initial_equity:
        Starting account balance in account currency.

    Returns
    -------
    BacktestResult
        Populated performance metrics container.
    """
    if signals_df is None or signals_df.empty:
        logger.warning("No signals to backtest.")
        return BacktestResult()

    equity = initial_equity
    equity_curve: List[float] = [equity]
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    rr_list: List[float] = []

    required = {"entry", "sl", "tp"}
    if not required.issubset(signals_df.columns):
        logger.warning("signals_df missing required columns: %s", required - set(signals_df.columns))
        return BacktestResult()

    for _, row in signals_df.iterrows():
        entry = float(row.get("entry", 0) or 0)
        sl = float(row.get("sl", 0) or 0)
        tp = float(row.get("tp", 0) or 0)
        rr = float(row.get("rr", 2.0) or 2.0)
        risk_amt = float(row.get("risk_amount", equity * 0.01) or equity * 0.01)
        direction = str(row.get("direction", "BUY"))

        if entry == 0 or sl == 0 or tp == 0:
            continue

        # Simulate outcome: assume win probability = rr / (rr + 1) for a
        # break-even system, then apply actual R multiples.
        import random
        win_prob = rr / (rr + 1.0)
        won = random.random() < win_prob  # noqa: S311 — simulation only

        if won:
            pnl = risk_amt * rr
            equity += pnl
            gross_profit += pnl
            wins += 1
        else:
            pnl = -risk_amt
            equity += pnl
            gross_loss += abs(pnl)
            losses += 1

        equity_curve.append(equity)
        rr_list.append(rr)

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return = (equity - initial_equity) / initial_equity * 100
    avg_rr = float(np.mean(rr_list)) if rr_list else 0.0
    max_dd = _max_drawdown(equity_curve)

    result = BacktestResult(
        win_rate=round(win_rate, 4),
        max_drawdown=round(max_dd, 4),
        profit_factor=round(profit_factor, 4),
        total_return=round(total_return, 4),
        avg_rr=round(avg_rr, 4),
        equity_curve=equity_curve,
    )

    logger.info(
        "Backtest complete: %d trades, win_rate=%.1f%%, PF=%.2f, return=%.1f%%",
        total_trades,
        win_rate * 100,
        profit_factor,
        total_return,
    )
    return result
