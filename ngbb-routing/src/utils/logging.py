"""Structured logging setup using Rich for console output.

Provides a consistent logging interface across all NGBB modules
with structured formatting, colored output, and optional file logging.
"""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_console = Console(stderr=True)


def setup_logger(
    name: str = "ngbb",
    level: str = "INFO",
    log_file: str | Path | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Create and configure a logger with Rich formatting.

    Args:
        name: Logger name (usually module name).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to file.
        rich_tracebacks: Whether to use Rich for traceback formatting.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler with Rich
    console_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=True,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=True,
        markup=True,
    )
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler (plain text)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt=_DATE_FORMAT,
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ngbb") -> logging.Logger:
    """Get an existing logger by name, or create one with defaults.

    Args:
        name: Logger name.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
