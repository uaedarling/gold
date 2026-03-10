"""
risk_engine/position_sizing.py — Account risk and lot-size calculation utilities.

Translates a percentage-based risk preference and account balance into a
concrete risk amount in account currency so that the TP/SL calculator can
derive position size.
"""

import logging

logger = logging.getLogger(__name__)


def risk_amount(account_balance: float, risk_pct: float) -> float:
    """Calculate the monetary risk for a single trade.

    Parameters
    ----------
    account_balance:
        Total account equity in the account's base currency.
    risk_pct:
        Percentage of equity to risk on this trade (e.g. ``1.0`` for 1 %).

    Returns
    -------
    float
        Risk in account currency (e.g. USD), rounded to two decimal places.

    Examples
    --------
    >>> risk_amount(10000, 1.0)
    100.0
    >>> risk_amount(5000, 0.5)
    25.0
    """
    if account_balance <= 0:
        raise ValueError(f"account_balance must be positive, got {account_balance}")
    if risk_pct <= 0:
        raise ValueError(f"risk_pct must be positive, got {risk_pct}")

    amount = round(account_balance * (risk_pct / 100.0), 2)
    logger.debug(
        "Risk amount: balance=%.2f, risk_pct=%.2f%% → %.2f",
        account_balance,
        risk_pct,
        amount,
    )
    return amount
