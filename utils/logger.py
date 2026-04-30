"""
utils/logger.py
---------------
Centralized, color-coded, timestamped logger for the entire system.
Import via:  from utils.logger import get_logger
"""

import io
import logging
import sys

# Force UTF-8 output on Windows so Unicode chars don't crash cp1252
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

import config

_LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

_FMT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with colour support if available."""
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured
        return logger

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.DEBUG)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if _HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + _FMT,
            datefmt=_DATE_FMT,
            log_colors=_LOG_COLORS,
        )
    else:
        formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
