"""
dashboard/web_dashboard.py — Flask + Plotly live monitoring dashboard.

Serves a single-page HTML dashboard that auto-refreshes every 15 seconds,
showing the current market state, AI predictions, and a history chart of all
generated signals.
"""

import logging
import os
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask

from gold_quant_ai.config import DASHBOARD_REFRESH_SECONDS

logger = logging.getLogger(__name__)

_SIGNALS_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals_log.csv")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_signals() -> pd.DataFrame:
    """Load the signals log CSV into a DataFrame.

    Returns an empty DataFrame if the file does not exist yet.

    Returns
    -------
    pd.DataFrame
        DataFrame of historical signals, or empty DataFrame when unavailable.
    """
    if not os.path.isfile(_SIGNALS_LOG):
        return pd.DataFrame()
    try:
        df = pd.read_csv(_SIGNALS_LOG, parse_dates=["timestamp"])
        return df
    except Exception as exc:
        logger.warning("Failed to load signals log: %s", exc)
        return pd.DataFrame()


def _signals_plot(df: pd.DataFrame) -> str:
    """Build a Plotly scatter chart of signal entry prices coloured by direction.

    Parameters
    ----------
    df:
        Signals DataFrame with at minimum ``timestamp``, ``entry``, and
        ``direction`` columns.

    Returns
    -------
    str
        HTML ``<div>`` string containing the interactive Plotly chart, or an
        empty string when *df* is empty or missing required columns.
    """
    if df.empty or not {"timestamp", "entry", "direction"}.issubset(df.columns):
        return "<p>No signal history yet.</p>"

    buy_df = df[df["direction"] == "BUY"]
    sell_df = df[df["direction"] == "SELL"]

    fig = go.Figure()

    if not buy_df.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_df["timestamp"],
                y=buy_df["entry"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="BUY",
            )
        )

    if not sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_df["timestamp"],
                y=sell_df["entry"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="SELL",
            )
        )

    fig.update_layout(
        title="Signal Entry Prices",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (XAUUSD)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_app(shared_state: Dict) -> Flask:
    """Create and configure the Flask dashboard application.

    Parameters
    ----------
    shared_state:
        Mutable dictionary populated by the analysis loop thread containing
        keys: ``last_run_utc``, ``market_bias``, ``trend_h4``, ``trend_h1``,
        ``latest_signal``, ``ai``, ``last_error``.

    Returns
    -------
    Flask
        Configured Flask application instance ready to run.
    """
    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)  # quiet Flask's own logger

    @app.route("/")
    def index():
        """Render the main dashboard page."""
        state = shared_state
        signal = state.get("latest_signal", {})
        ai = state.get("ai", {})
        signals_df = _load_signals()
        chart_html = _signals_plot(signals_df)

        bias = state.get("market_bias", "NEUTRAL")
        bias_colour = "#2ecc71" if bias == "BULLISH" else "#e74c3c" if bias == "BEARISH" else "#f39c12"

        direction = signal.get("direction", "N/A")
        dir_colour = "#2ecc71" if direction == "BUY" else "#e74c3c" if direction == "SELL" else "#aaa"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta http-equiv="refresh" content="{DASHBOARD_REFRESH_SECONDS}"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GOLD_QUANT_AI Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #0d1117; color: #c9d1d9;
      padding: 20px;
    }}
    h1 {{ color: #f0b429; margin-bottom: 20px; font-size: 1.8rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
    }}
    .card h3 {{ color: #8b949e; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px; }}
    .card .value {{ font-size: 1.5rem; font-weight: bold; }}
    .chart-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    .error {{ color: #ff6b6b; font-size: 0.8rem; margin-top: 8px; }}
    footer {{ margin-top: 20px; font-size: 0.75rem; color: #484f58; }}
  </style>
</head>
<body>
  <h1>⚡ GOLD_QUANT_AI Dashboard</h1>

  <div class="grid">
    <div class="card">
      <h3>Market Bias (H4)</h3>
      <div class="value" style="color:{bias_colour}">{bias}</div>
    </div>
    <div class="card">
      <h3>H4 Trend</h3>
      <div class="value">{state.get('trend_h4', 'UNKNOWN')}</div>
    </div>
    <div class="card">
      <h3>H1 Trend</h3>
      <div class="value">{state.get('trend_h1', 'UNKNOWN')}</div>
    </div>
    <div class="card">
      <h3>AI Confidence</h3>
      <div class="value">{ai.get('confidence', 0.0):.1%}</div>
    </div>
    <div class="card">
      <h3>Prob Up / Down</h3>
      <div class="value">
        <span style="color:#2ecc71">{ai.get('prob_up', 0.5):.1%}</span>
        &nbsp;/&nbsp;
        <span style="color:#e74c3c">{ai.get('prob_down', 0.5):.1%}</span>
      </div>
    </div>
    <div class="card">
      <h3>Latest Signal</h3>
      <div class="value" style="color:{dir_colour}">{direction}</div>
    </div>
    <div class="card">
      <h3>Entry / SL / TP</h3>
      <div class="value" style="font-size:1.1rem">
        {signal.get('entry', '—')} /
        {signal.get('sl', '—')} /
        {signal.get('tp', '—')}
      </div>
    </div>
    <div class="card">
      <h3>Last Run (UTC)</h3>
      <div class="value" style="font-size:0.95rem">{state.get('last_run_utc', 'Never')}</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h3>Signal Reason</h3>
    <p style="margin-top:8px">{signal.get('reason', '—')}</p>
    {"<p class='error'>Last error: " + state.get('last_error','') + "</p>" if state.get('last_error') else ""}
  </div>

  <div class="chart-card">
    <h3 style="color:#8b949e; font-size:0.75rem; text-transform:uppercase; margin-bottom:12px">Signal History</h3>
    {chart_html}
  </div>

  <footer>Auto-refreshes every {DASHBOARD_REFRESH_SECONDS} seconds &bull; {len(signals_df)} signals logged</footer>
</body>
</html>"""
        return html

    return app
