"""
Logging configuration for the Real-time Translator.

Call ``setup_logging()`` once at application startup (in ``main()``).
Every module then retrieves its own logger with::

    import logging
    logger = logging.getLogger(__name__)

Log levels (low → high):
  DEBUG    — verbose per-chunk diagnostics
  INFO     — normal operational messages (model loading, detections, …)
  WARNING  — non-fatal problems (fallback model used, quiet audio, …)
  ERROR    — failures that stop a single chunk from being processed
  CRITICAL — fatal errors (unused in this codebase)

The default level is INFO.  Pass ``level="DEBUG"`` to see per-chunk details.
"""

import logging
import sys
from typing import Optional

# Default format: timestamp [LEVEL] module.name: message
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` (case-
        insensitive).  Invalid strings are silently treated as ``"INFO"``.
    log_file:
        Optional path to a log file.  When set, log records are written to
        *both* stderr and the file.  The file is opened in append mode so
        multiple runs are concatenated.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
    )
