"""Logging helpers for LightSwapConverter.

Only the standard library is used so the logging layer works on a clean
offline Windows install. A rotating file handler keeps disk usage bounded on
machines with small drives.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

LOG_FILENAME = "lightswapconverter.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 512 * 1024
BACKUP_COUNT = 2

_configured = False


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """Configure the root logger once and return the application logger.

    Calling this more than once is safe: handlers are only attached on the
    first call, later calls just adjust the level.
    """
    global _configured

    logger = logging.getLogger("lightswapconverter")
    numeric_level = _coerce_level(level)
    logger.setLevel(numeric_level)

    if _configured:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / LOG_FILENAME,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)
        except OSError:
            # A read-only or missing log directory must never stop the app.
            logger.addHandler(logging.NullHandler())

    if console and sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        logger.addHandler(stream_handler)

    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("core.processor")``."""
    if not name or name == "lightswapconverter":
        return logging.getLogger("lightswapconverter")
    return logging.getLogger(f"lightswapconverter.{name}")


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO
