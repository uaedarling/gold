"""
data_engine/data_storage.py — CSV persistence utilities for GOLD_QUANT_AI.

Provides thin wrappers around pandas CSV I/O together with directory
creation helpers so that the rest of the codebase does not need to
deal with filesystem concerns directly.
"""

import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if they do not exist.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
    logger.debug("Directory ensured: %s", path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Persist a DataFrame to a CSV file.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination file path (absolute or relative).
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to '%s'.", len(df), path)


def load_csv(path: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV file.

    Parameters
    ----------
    path:
        Source file path (absolute or relative).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If no file exists at *path*.
    """
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from '%s'.", len(df), path)
    return df
