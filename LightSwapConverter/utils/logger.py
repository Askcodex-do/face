"""Logging helpers for LightSwapConverter.

The GUI needs to display log lines while the core stays headless, so the module
provides a single shared logger plus an in-memory handler that the interface can
attach to. Nothing here touches the network; logs go to stderr and, optionally,
to a size limited rotating file inside the user's home directory.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional

from .config import LoggingConfig, default_config_path

LOGGER_NAME = "lightswapconverter"

_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def default_log_path(file_name: str = "lightswapconverter.log") -> Path:
    """Return the log file location, next to the configuration file."""
    return default_config_path().parent / file_name


class MemoryHandler(logging.Handler):
    """Keep the most recent records in a bounded deque.

    The GUI polls :meth:`records` on a timer, which avoids any threading
    concerns around Tkinter widgets.
    """

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._records: Deque[str] = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            self._records.append(self.format(record))
        except Exception:  # pragma: no cover - never break the app on logging
            pass

    def records(self) -> List[str]:
        """Return the buffered lines, oldest first."""
        return list(self._records)

    def drain(self) -> List[str]:
        """Return the buffered lines and clear the buffer."""
        lines = list(self._records)
        self._records.clear()
        return lines


_memory_handler: Optional[MemoryHandler] = None


def setup_logging(
    config: Optional[LoggingConfig] = None,
    *,
    log_file: Optional[Path] = None,
    stream: bool = True,
) -> logging.Logger:
    """Configure and return the shared application logger.

    Calling this more than once is safe: existing handlers installed by this
    function are removed first so repeated calls do not duplicate output.
    """
    global _memory_handler

    config = config or LoggingConfig()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    # The application owns its own handlers; do not double log via the root.
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    if stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if config.file_name:
        target = Path(log_file) if log_file is not None else default_log_path(
            config.file_name
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                target,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # A read only or missing directory must not stop the application.
            logger.warning("file logging disabled, cannot write to %s", target)

    _memory_handler = MemoryHandler()
    logger.addHandler(_memory_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the application logger.

    ``get_logger("core.video_reader")`` yields the logger used by that module.
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def get_memory_handler() -> Optional[MemoryHandler]:
    """Return the buffered handler, or ``None`` if logging is not set up yet."""
    return _memory_handler


def configure_for_tests(level: str = "DEBUG") -> logging.Logger:
    """Set up logging without touching the filesystem. Used by the test suite."""
    config = LoggingConfig(level=level, file_name="")
    return setup_logging(config, stream=False)


def iter_log_lines(limit: int = 100) -> Iterable[str]:
    """Convenience helper returning at most ``limit`` recent log lines."""
    handler = get_memory_handler()
    if handler is None:
        return []
    lines = handler.records()
    return lines[-limit:]
